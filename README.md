# IA Finance

This repository contains a simple example of a machine learning model for financial time series.

The script `finance_model.py` trains a linear regression model to predict the next closing price of a
stock using lagged values. The code attempts to download real data using the `yfinance` package, but
falls back to generating synthetic data if the download fails (for example in offline environments).

## Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
- `yfinance` (optional, for fetching live data)

All dependencies can be installed with `pip install -r requirements.txt`. A `requirements.txt` file is
provided.

## Usage

Run the model script:

```bash
python finance_model.py
```

The script prints the mean squared error on a hold-out set and shows the last five predictions
compared to their actual values.

## Notes

This example is intentionally simple and uses a linear regression model with three lagged features.
For real-world applications, more sophisticated models and data handling would be necessary.
