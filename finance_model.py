import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf


def generate_synthetic_data(size=500, seed=0):
    """Generate a synthetic price series using a simple random walk."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0, scale=1, size=size)
    prices = 100 + np.cumsum(steps)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=size)
    return pd.DataFrame({"Close": prices}, index=dates)


def load_data(ticker="AAPL", period="2y"):
    """Download price data; fall back to synthetic data on failure."""
    try:
        data = yf.download(ticker, period=period, progress=False)
        data = data[["Close"]]
        data.dropna(inplace=True)
        if len(data) == 0:
            raise ValueError("No data returned")
        return data
    except Exception as exc:
        print(f"Data download failed: {exc}. Using synthetic data instead.")
        return generate_synthetic_data()


def build_features(data, lags=3):
    df = data.copy()
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df.dropna(inplace=True)
    X = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y = df["Close"].values
    return X, y


def train_test_split_time_series(X, y, test_size=20):
    if len(X) <= test_size:
        raise ValueError("Not enough data for the test size")
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    return mse, predictions


def main():
    data = load_data()
    X, y = build_features(data)
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)
    model = train_model(X_train, y_train)
    mse, predictions = evaluate(model, X_test, y_test)
    print("Mean Squared Error:", mse)
    print("Last 5 Predictions vs Actuals:")
    for pred, actual in zip(predictions[-5:], y_test[-5:]):
        print(f"pred={pred:.2f}, actual={actual:.2f}")


if __name__ == "__main__":
    main()
