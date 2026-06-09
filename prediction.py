import yfinance as yf
import pandas as pd
import numpy as np

from yahooquery import Ticker
from xgboost import XGBClassifier

start_date = "2015-01-01"
future_days = 126
top_n = 10
max_per_sector = 3

tickers = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","NFLX",
    "JPM","V","MA","COST","WMT","HD","PG","KO","PEP","MCD",
    "XOM","CVX","COP","UNH","LLY","ABBV","JNJ","MRK","PFE","TMO",
    "CRM","ORCL","ADBE","CSCO","INTC","QCOM","TXN","IBM","NOW","AMAT",
    "CAT","BA","GE","HON","UPS","LMT","RTX","DE","MMM","LOW"
]

features = [
    "Return_5d", "Return_20d", "Return_60d", "Return_252d",
    "MA50_vs_MA200", "Volatility_20d", "Volume_Change",
    "Relative_Strength_20d", "Relative_Strength_60d",
    "RSI", "MACD_Diff",
    "Market_Cap", "Forward_PE", "Trailing_PE", "Price_To_Book",
    "Profit_Margins", "Revenue_Growth", "Earnings_Growth",
    "Debt_To_Equity", "ROE", "Free_Cashflow"
]


def get_series(data, column):
    result = data[column]
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    return result


def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]

    return df


def get_fundamentals(tickers):
    print("\nDownloading fundamentals...")

    tq = Ticker(tickers)

    summary = tq.summary_detail
    financial = tq.financial_data
    profile = tq.asset_profile

    rows = []

    for ticker in tickers:
        s = summary.get(ticker, {}) if isinstance(summary, dict) else {}
        f = financial.get(ticker, {}) if isinstance(financial, dict) else {}
        p = profile.get(ticker, {}) if isinstance(profile, dict) else {}

        rows.append({
            "Ticker": ticker,
            "Sector": p.get("sector", "Unknown"),
            "Market_Cap": s.get("marketCap", np.nan),
            "Forward_PE": s.get("forwardPE", np.nan),
            "Trailing_PE": s.get("trailingPE", np.nan),
            "Price_To_Book": s.get("priceToBook", np.nan),
            "Profit_Margins": f.get("profitMargins", np.nan),
            "Revenue_Growth": f.get("revenueGrowth", np.nan),
            "Earnings_Growth": f.get("earningsGrowth", np.nan),
            "Debt_To_Equity": f.get("debtToEquity", np.nan),
            "ROE": f.get("returnOnEquity", np.nan),
            "Free_Cashflow": f.get("freeCashflow", np.nan),
        })

    return pd.DataFrame(rows)


fundamentals = get_fundamentals(tickers)

print("Downloading SPY...")

spy = yf.download(
    "SPY",
    start=start_date,
    auto_adjust=True,
    progress=False
)

spy_close = get_series(spy, "Close")

all_data = []
live_data = []

for ticker in tickers:
    try:
        print(f"Downloading {ticker}...")

        stock = yf.download(
            ticker,
            start=start_date,
            auto_adjust=True,
            progress=False
        )

        if stock.empty:
            print(f"Skipping {ticker}")
            continue

        close = get_series(stock, "Close")
        volume = get_series(stock, "Volume")

        df = pd.DataFrame(index=stock.index)

        df["Ticker"] = ticker
        df["Close"] = close
        df["Volume"] = volume
        df["SPY_Close"] = spy_close.reindex(df.index)

        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_20d"] = df["Close"].pct_change(20)
        df["Return_60d"] = df["Close"].pct_change(60)
        df["Return_252d"] = df["Close"].pct_change(252)

        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()
        df["MA50_vs_MA200"] = df["MA50"] / df["MA200"] - 1

        df["Daily_Return"] = df["Close"].pct_change()
        df["Volatility_20d"] = df["Daily_Return"].rolling(20).std()

        df["Volume_Change"] = (
            df["Volume"]
            / df["Volume"].rolling(20).mean()
            - 1
        )

        df["SPY_Return_20d"] = df["SPY_Close"].pct_change(20)
        df["SPY_Return_60d"] = df["SPY_Close"].pct_change(60)

        df["Relative_Strength_20d"] = (
            df["Return_20d"] - df["SPY_Return_20d"]
        )

        df["Relative_Strength_60d"] = (
            df["Return_60d"] - df["SPY_Return_60d"]
        )

        df = add_rsi(df)
        df = add_macd(df)

        df["Future_Return"] = (
            df["Close"].shift(-future_days)
            / df["Close"]
            - 1
        )

        # Keep latest rows for live prediction
        live_data.append(df.copy())

        # Training rows only where future is known
        train_df = df.dropna(subset=[
            "Future_Return",
            "Return_5d",
            "Return_20d",
            "Return_60d",
            "Return_252d",
            "MA50_vs_MA200",
            "Volatility_20d",
            "Volume_Change",
            "Relative_Strength_20d",
            "Relative_Strength_60d",
            "RSI",
            "MACD_Diff"
        ]).copy()

        all_data.append(train_df)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")


# -----------------------------
# TRAINING DATA
# -----------------------------

data = pd.concat(all_data).sort_index()

data["Date"] = data.index

data = data.merge(
    fundamentals,
    on="Ticker",
    how="left"
)

data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index("Date")
data = data.sort_index()

data["Future_Return_Rank"] = (
    data.groupby(data.index)["Future_Return"].rank(pct=True)
)

data["Top_20"] = (
    data["Future_Return_Rank"] >= 0.80
).astype(int)

print("\nTarget distribution:")
print(data["Top_20"].value_counts())


# -----------------------------
# LIVE DATA
# -----------------------------

live_data = pd.concat(live_data).sort_index()

live_data["Date"] = live_data.index

live_data = live_data.merge(
    fundamentals,
    on="Ticker",
    how="left"
)

live_data["Date"] = pd.to_datetime(live_data["Date"])
live_data = live_data.set_index("Date")
live_data = live_data.sort_index()


# -----------------------------
# CLEAN FEATURES
# -----------------------------

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    live_data[col] = pd.to_numeric(live_data[col], errors="coerce")

    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    live_data[col] = live_data[col].replace([np.inf, -np.inf], np.nan)

    median_value = data[col].median()

    data[col] = data[col].fillna(median_value)
    live_data[col] = live_data[col].fillna(median_value)


X = data[features]
y = data["Top_20"]

if y.nunique() < 2:
    raise ValueError("Target only has one class. Need both 0 and 1.")


# -----------------------------
# TRAIN MODEL
# -----------------------------

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X, y)


# -----------------------------
# CURRENT RANKING USING LATEST DATA
# -----------------------------

latest_rows = []

for ticker in live_data["Ticker"].unique():

    ticker_data = live_data[live_data["Ticker"] == ticker]

    latest = ticker_data.tail(1)

    if latest.empty:
        continue

    probability = model.predict_proba(latest[features])[0][1]

    sector = latest["Sector"].values[0]

    latest_rows.append({
        "Ticker": ticker,
        "Sector": sector,
        "Probability_Top_20": probability,
        "Latest_Date": latest.index[0],
        "Return_252d": latest["Return_252d"].values[0],
        "Volatility_20d": latest["Volatility_20d"].values[0],
        "Forward_PE": latest["Forward_PE"].values[0],
        "Revenue_Growth": latest["Revenue_Growth"].values[0],
        "Earnings_Growth": latest["Earnings_Growth"].values[0],
        "ROE": latest["ROE"].values[0],
    })


print(f"\nLive predictions created: {len(latest_rows)}")

if len(latest_rows) == 0:
    raise ValueError("No live predictions were created.")

ranking = pd.DataFrame(latest_rows)

ranking = ranking.sort_values(
    by="Probability_Top_20",
    ascending=False
)


# -----------------------------
# DIVERSIFIED PORTFOLIO
# -----------------------------

sector_counts = {}
portfolio = []

for _, row in ranking.iterrows():
    sector = row["Sector"]

    if sector_counts.get(sector, 0) >= max_per_sector:
        continue

    portfolio.append(row)
    sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(portfolio) == top_n:
        break

portfolio_df = pd.DataFrame(portfolio)


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)


# -----------------------------
# OUTPUT
# -----------------------------

print("\nVersion 6 Current Top 20 Ranking Using Latest Data:")

for _, row in ranking.head(20).iterrows():
    print(
        f"{row['Ticker']}: "
        f"{row['Probability_Top_20']:.2%} "
        f"- {row['Sector']} "
        f"- Date: {row['Latest_Date'].date()}"
    )


print("\nDiversified Top 10 Portfolio:")

for _, row in portfolio_df.iterrows():
    print(
        f"{row['Ticker']}: "
        f"{row['Probability_Top_20']:.2%} "
        f"- {row['Sector']}"
    )


print("\nFeature Importance:")
print(importance)


ranking.to_csv("version6_current_ranking.csv", index=False)
portfolio_df.to_csv("version6_diversified_portfolio.csv", index=False)
importance.to_csv("version6_feature_importance.csv", index=False)

print("\nSaved files:")
print("version6_current_ranking.csv")
print("version6_diversified_portfolio.csv")
print("version6_feature_importance.csv")