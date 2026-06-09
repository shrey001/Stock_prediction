import yfinance as yf
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

start_date = "2015-01-01"
future_days = 126

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
spy = yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
spy_close = get_series(spy, "Close")

all_data = []

for ticker in tickers:
    try:
        print(f"Downloading {ticker}...")

        stock = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)

        if stock.empty:
            print(f"Skipping {ticker}")
            continue

        close = get_series(stock, "Close")
        volume = get_series(stock, "Volume")

        df = pd.DataFrame(index=stock.index)

        df["Ticker"] = ticker
        df["Close"] = close
        df["Volume"] = volume
        df["SPY_Close"] = spy_close

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

        df.dropna(inplace=True)

        all_data.append(df)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")


data = pd.concat(all_data)
data = data.sort_index()

# Top 20% target by date
data["Future_Return_Rank"] = data.groupby(data.index)["Future_Return"].rank(pct=True)

data["Top_20"] = (data["Future_Return_Rank"] >= 0.80).astype(int)

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

X = data[features]
y = data["Top_20"]

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

latest_rows = []

for ticker in data["Ticker"].unique():
    ticker_data = data[data["Ticker"] == ticker]

    latest = ticker_data.tail(1)
    latest_X = latest[features]

    probability = model.predict_proba(latest_X)[0][1]

    latest_rows.append({
        "Ticker": ticker,
        "Probability_Top_20": probability
    })

ranking = pd.DataFrame(latest_rows)

ranking = ranking.sort_values(
    by="Probability_Top_20",
    ascending=False
)

print("\nTop 20 Predicted Stocks:\n")

for _, row in ranking.head(20).iterrows():
    ticker = row["Ticker"]
    probability = row["Probability_Top_20"]

    if probability > 0.30:
        signal = "STRONG"
    elif probability > 0.20:
        signal = "WATCH"
    else:
        signal = "WEAK"

    print(f"{ticker}: {probability:.2%} - {signal}")

ranking.to_csv("top20_stock_ranking.csv", index=False)

print("\nSaved ranking to top20_stock_ranking.csv")