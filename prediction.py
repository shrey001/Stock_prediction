import time
import yfinance as yf
import pandas as pd
import numpy as np

from yahooquery import Ticker
from xgboost import XGBClassifier

start_date = "2015-01-01"
future_days = 126
top_n = 10
max_per_sector = 3
ticker_file = "Tickers.csv"

features = [
    "Return_5d", "Return_20d", "Return_60d", "Return_252d",
    "MA50_vs_MA200", "Volatility_20d", "Volume_Change",
    "Relative_Strength_20d", "Relative_Strength_60d",
    "RSI", "MACD_Diff",
    "Market_Cap", "Forward_PE", "Trailing_PE", "Price_To_Book",
    "Profit_Margins", "Revenue_Growth", "Earnings_Growth",
    "Debt_To_Equity", "ROE", "Free_Cashflow"
]


def get_stock_universe():
    print("Loading tickers from tickers.csv...")

    df = pd.read_csv(ticker_file)

    if "Ticker" not in df.columns:
        raise ValueError("tickers.csv must have a column named Ticker")

    tickers = df["Ticker"].dropna().astype(str).tolist()

    tickers = [
        t.replace(".", "-").strip().upper()
        for t in tickers
        if t.strip() != ""
    ]

    tickers = sorted(list(set(tickers)))

    return tickers


def download_with_retry(ticker, start_date, max_retries=5):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} downloading {ticker}...")

            data = yf.download(
                ticker,
                start=start_date,
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if not data.empty:
                return data

            print(f"{ticker} returned empty data.")

        except Exception as e:
            print(f"{ticker} download error: {e}")

        sleep_time = 30 * (attempt + 1)
        print(f"Waiting {sleep_time} seconds before retry...")
        time.sleep(sleep_time)

    print(f"Failed to download {ticker} after {max_retries} attempts.")
    return pd.DataFrame()


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


def safe_get(source, ticker):
    if not isinstance(source, dict):
        return {}

    value = source.get(ticker, {})

    if not isinstance(value, dict):
        return {}

    return value


def get_fundamentals(tickers, batch_size=50):
    print("\nDownloading fundamentals...")

    rows = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        print(f"Fundamentals batch {i + 1} to {i + len(batch)}")

        try:
            tq = Ticker(batch)

            summary = tq.summary_detail
            financial = tq.financial_data
            profile = tq.asset_profile

        except Exception as e:
            print(f"Could not download fundamentals batch: {e}")
            summary = {}
            financial = {}
            profile = {}

        for ticker in batch:
            s = safe_get(summary, ticker)
            f = safe_get(financial, ticker)
            p = safe_get(profile, ticker)

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

        time.sleep(2)

    return pd.DataFrame(rows)


tickers = get_stock_universe()

print(f"Total stocks in universe: {len(tickers)}")

if len(tickers) == 0:
    raise ValueError("No tickers found in tickers.csv")

fundamentals = get_fundamentals(tickers)

print("Downloading SPY...")

spy = download_with_retry("SPY", start_date, max_retries=6)

if spy.empty:
    raise ValueError("SPY download failed after retries. Cannot calculate relative strength.")

spy_close = get_series(spy, "Close")

all_data = []
live_data = []

for index, ticker in enumerate(tickers, start=1):
    try:
        print(f"Downloading {index}/{len(tickers)}: {ticker}...")

        stock = download_with_retry(
            ticker,
            start_date,
            max_retries=3
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

        live_data.append(df.copy())

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

        if not train_df.empty:
            all_data.append(train_df)

        time.sleep(1)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")


if len(all_data) == 0:
    raise ValueError("No training data created.")

if len(live_data) == 0:
    raise ValueError("No live data created.")


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


for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    live_data[col] = pd.to_numeric(live_data[col], errors="coerce")

    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    live_data[col] = live_data[col].replace([np.inf, -np.inf], np.nan)

    median_value = data[col].median()

    if pd.isna(median_value):
        median_value = 0

    data[col] = data[col].fillna(median_value)
    live_data[col] = live_data[col].fillna(median_value)


X = data[features]
y = data["Top_20"]

if y.nunique() < 2:
    raise ValueError("Target only has one class. Need both 0 and 1.")


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


importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)


print("\nCurrent Top 20 Ranking Using Latest Data:")

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
