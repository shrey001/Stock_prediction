import yfinance as yf
import pandas as pd

from xgboost import XGBClassifier

start_date = "2015-01-01"
future_days = 126
top_n = 10
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
    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD_Diff"] = (
        df["MACD"]
        - df["MACD_Signal"]
    )

    return df


print("Downloading SPY...")

spy = yf.download(
    "SPY",
    start=start_date,
    auto_adjust=True,
    progress=False
)

spy_close = get_series(
    spy,
    "Close"
)

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

        close = get_series(
            stock,
            "Close"
        )

        volume = get_series(
            stock,
            "Volume"
        )

        df = pd.DataFrame(
            index=stock.index
        )

        df["Ticker"] = ticker
        df["Close"] = close
        df["Volume"] = volume
        df["SPY_Close"] = spy_close.reindex(df.index)

        # Features
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_20d"] = df["Close"].pct_change(20)
        df["Return_60d"] = df["Close"].pct_change(60)
        df["Return_252d"] = df["Close"].pct_change(252)

        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        df["MA50_vs_MA200"] = (
            df["MA50"]
            / df["MA200"]
            - 1
        )

        df["Daily_Return"] = (
            df["Close"]
            .pct_change()
        )

        df["Volatility_20d"] = (
            df["Daily_Return"]
            .rolling(20)
            .std()
        )

        df["Volume_Change"] = (
            df["Volume"]
            / df["Volume"]
            .rolling(20)
            .mean()
            - 1
        )

        df["SPY_Return_20d"] = (
            df["SPY_Close"]
            .pct_change(20)
        )

        df["SPY_Return_60d"] = (
            df["SPY_Close"]
            .pct_change(60)
        )

        df["Relative_Strength_20d"] = (
            df["Return_20d"]
            - df["SPY_Return_20d"]
        )

        df["Relative_Strength_60d"] = (
            df["Return_60d"]
            - df["SPY_Return_60d"]
        )

        df = add_rsi(df)
        df = add_macd(df)

        # Future 6-month return
        df["Future_Return"] = (
            df["Close"]
            .shift(-future_days)
            / df["Close"]
            - 1
        )

        df["SPY_Future_Return"] = (
            df["SPY_Close"]
            .shift(-future_days)
            / df["SPY_Close"]
            - 1
        )

        df.dropna(inplace=True)

        all_data.append(df)

    except Exception as e:
        print(f"Skipping {ticker}: {e}")

data = pd.concat(all_data)
data = data.sort_index()

# Top 20% winners
data["Future_Return_Rank"] = (
    data.groupby(data.index)
    ["Future_Return"]
    .rank(pct=True)
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

all_dates = sorted(
    data.index.unique()
)

# Non-overlapping 6-month steps
rebalance_dates = all_dates[::future_days]

results = []

print("\nRunning NON-OVERLAPPING backtest...")

for test_date in rebalance_dates:

    train_data = data[
        data.index < test_date
    ]

    test_data = data[
        data.index == test_date
    ].copy()

    if (
        len(train_data) < 5000
        or len(test_data) < top_n
    ):
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

    model.fit(
        X_train,
        y_train
    )

    X_today = test_data[features]

    test_data["Prediction"] = (
        model.predict_proba(X_today)
        [:, 1]
    )

    picks = test_data.sort_values(
        by="Prediction",
        ascending=False
    ).head(top_n)

    portfolio_return = (
        picks["Future_Return"]
        .mean()
        - trading_cost
    )

    spy_return = (
        picks["SPY_Future_Return"]
        .mean()
    )

    results.append({
        "Date": test_date,
        "Portfolio_Return":
        portfolio_return,
        "SPY_Return":
        spy_return,
        "Top_Picks":
        ", ".join(
            picks["Ticker"]
        )
    })

results_df = pd.DataFrame(results)

avg_portfolio_return = (
    results_df[
        "Portfolio_Return"
    ].mean()
)

avg_spy_return = (
    results_df[
        "SPY_Return"
    ].mean()
)

win_rate = (
    results_df[
        "Portfolio_Return"
    ]
    >
    results_df[
        "SPY_Return"
    ]
).mean()

annualized_portfolio = (
    (1 + avg_portfolio_return)
    ** 2
    - 1
)

annualized_spy = (
    (1 + avg_spy_return)
    ** 2
    - 1
)

print("\nFinal Honest Results:")
print(
    f"Periods tested: "
    f"{len(results_df)}"
)
print(
    f"Portfolio avg "
    f"6-month return: "
    f"{avg_portfolio_return:.2%}"
)
print(
    f"SPY avg "
    f"6-month return: "
    f"{avg_spy_return:.2%}"
)
print(
    f"Win rate vs SPY: "
    f"{win_rate:.2%}"
)
print(
    f"Portfolio annualized: "
    f"{annualized_portfolio:.2%}"
)
print(
    f"SPY annualized: "
    f"{annualized_spy:.2%}"
)

print("\nMost recent picks:")
print(
    results_df.tail(1)
    ["Top_Picks"]
    .values[0]
)