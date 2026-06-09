import yfinance as yf
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Choose stock ticker
ticker = "AAPL"
ticker = "MSFT"
ticker = "NVDA"
ticker = "SPY"

# Download stock data
df = yf.download(ticker, start="2015-01-01", auto_adjust=True)

# Features
df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)
df["Return_20d"] = df["Close"].pct_change(20)

df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

df["Volatility"] = df["Return_1d"].rolling(20).std()

# Predict 6 months ahead (126 trading days)
df["Future_Return"] = (
    df["Close"].shift(-126) / df["Close"] - 1
)

# Remove missing rows
df.dropna(inplace=True)

# Features used by model
features = [
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "MA50",
    "MA200",
    "Volatility",
    "Volume"
]

X = df[features]
y = df["Future_Return"]

# Split train/test by time
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Train XGBoost
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)

print(f"Ticker: {ticker}")
print(f"Mean Absolute Error: {mae:.4f}")

# Latest prediction
latest = X.tail(1)
prediction = model.predict(latest)[0]

print(
    f"Predicted 6-month return: "
    f"{prediction:.2%}"
)