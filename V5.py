import yfinance as yf
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

start_date = "2015-01-01"
future_days = 126
top_n = 10
starting_cash = 10000
trading_cost = 0.002

tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "V", "MA", "COST", "WMT", "HD", "PG", "KO", "PEP", "MCD",
    "XOM", "CVX", "COP", "UNH", "LLY", "ABBV", "JNJ", "MRK", "PFE", "TMO",
    "CRM", "ORCL", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "IBM", "NOW", "AMAT",
    "CAT", "BA", "GE", "HON", "UPS", "LMT", "RTX", "DE", "MMM", "LOW"
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


print("Downloading SPY...")

spy = yf.download(
    "SPY",
    start=start_date,
    auto_adjust=True,
    progress=False
)

spy_close = get_series(spy, "Close")

all_data = []

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

        df["Volume_Change"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1

        df["SPY_Return_20d"] = df["SPY_Close"].pct_change(20)
        df["SPY_Return_60d"] = df["SPY_Close"].pct_change(60)

        df["Relative_Strength_20d"] = df["Return_20d"] - df["SPY_Return_20d"]
        df["Relative_Strength_60d"] = df["Return_60d"] - df["SPY_Return_60d"]

        df = add_rsi(df)
        df = add_macd(df)

        df["Future_Return"] = df["Close"].shift(-future_days) / df["Close"] - 1
        df["SPY_Future_Return"] = df["SPY_Close"].shift(-future_days) / df["SPY_Close"] - 1

        df.dropna(inplace=True)

        all_data.append(df)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")


data = pd.concat(all_data)
data = data.sort_index()

data["Future_Return_Rank"] = (
    data.groupby(data.index)["Future_Return"].rank(pct=True)
)

data["Top_20"] = (
    data["Future_Return_Rank"] >= 0.80
).astype(int)

features = [
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
]

all_dates = pd.Series(sorted(data.index.unique()))

monthly_dates = all_dates.groupby(
    [all_dates.dt.year, all_dates.dt.month]
).max().tolist()

results = []

print("\nRunning Version 5 backtest...")

for test_date in monthly_dates:

    train_data = data[data.index < test_date]
    test_data = data[data.index == test_date].copy()

    if len(train_data) < 5000 or len(test_data) < top_n:
        continue

    X_train = train_data[features]
    y_train = train_data["Top_20"]

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    test_data["Prediction"] = model.predict_proba(test_data[features])[:, 1]

    picks = test_data.sort_values(
        by="Prediction",
        ascending=False
    ).head(top_n)

    portfolio_return = picks["Future_Return"].mean() - trading_cost
    spy_return = picks["SPY_Future_Return"].mean()

    results.append({
        "Date": test_date,
        "Portfolio_6M_Return": portfolio_return,
        "SPY_6M_Return": spy_return,
        "Top_Picks": ", ".join(picks["Ticker"].tolist())
    })


results_df = pd.DataFrame(results)

avg_portfolio_6m = results_df["Portfolio_6M_Return"].mean()
avg_spy_6m = results_df["SPY_6M_Return"].mean()

win_rate = (
    results_df["Portfolio_6M_Return"] > results_df["SPY_6M_Return"]
).mean()

annualized_portfolio = (1 + avg_portfolio_6m) ** 2 - 1
annualized_spy = (1 + avg_spy_6m) ** 2 - 1

portfolio_value = starting_cash
spy_value = starting_cash

equity_curve = []

for _, row in results_df.iterrows():
    portfolio_value *= (1 + row["Portfolio_6M_Return"])
    spy_value *= (1 + row["SPY_6M_Return"])

    equity_curve.append({
        "Date": row["Date"],
        "AI_Portfolio_Value": portfolio_value,
        "SPY_Value": spy_value
    })

equity_df = pd.DataFrame(equity_curve)

print("\nVersion 5 Results:")
print(f"Periods tested: {len(results_df)}")
print(f"Average portfolio 6-month return: {avg_portfolio_6m:.2%}")
print(f"Average SPY 6-month return: {avg_spy_6m:.2%}")
print(f"Win rate vs SPY: {win_rate:.2%}")
print(f"Estimated portfolio annualized return: {annualized_portfolio:.2%}")
print(f"Estimated SPY annualized return: {annualized_spy:.2%}")

print("\nPortfolio Growth:")
print(f"AI Portfolio final value: ${portfolio_value:,.2f}")
print(f"SPY final value: ${spy_value:,.2f}")

print("\nMost recent picks:")
print(results_df.tail(1)["Top_Picks"].values[0])

# Train final model on all data for current ranking
final_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

final_model.fit(data[features], data["Top_20"])

importance = pd.DataFrame({
    "Feature": features,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

latest_rows = []

for ticker in data["Ticker"].unique():
    ticker_data = data[data["Ticker"] == ticker]
    latest = ticker_data.tail(1)

    probability = final_model.predict_proba(latest[features])[0][1]

    latest_rows.append({
        "Ticker": ticker,
        "Probability_Top_20": probability
    })

ranking = pd.DataFrame(latest_rows)
ranking = ranking.sort_values(
    by="Probability_Top_20",
    ascending=False
)

print("\nCurrent Top 20 Ranking:")
for _, row in ranking.head(20).iterrows():
    print(f"{row['Ticker']}: {row['Probability_Top_20']:.2%}")

results_df.to_csv("version5_backtest_results.csv", index=False)
equity_df.to_csv("version5_equity_curve.csv", index=False)
ranking.to_csv("version5_current_ranking.csv", index=False)
importance.to_csv("version5_feature_importance.csv", index=False)

print("\nSaved files:")
print("version5_backtest_results.csv")
print("version5_equity_curve.csv")
print("version5_current_ranking.csv")
print("version5_feature_importance.csv")