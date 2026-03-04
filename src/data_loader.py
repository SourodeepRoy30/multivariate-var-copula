"""
data_loader.py
==============
Utilities for downloading and preprocessing asset price data.

Assets used in this project:
    - FTSE 100       (^FTSE)
    - GBP/USD        (GBPUSD=X)
    - Brent Crude    (BZ=F)

All data sourced from Yahoo Finance via yfinance.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ── Constants ────────────────────────────────────────────────────────────────

TICKERS = {
    "FTSE100": "^FTSE",
    "GBPUSD": "GBPUSD=X",
    "Brent": "BZ=F",
}

TRAIN_START = "2010-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2024-12-31"


# ── Data Download ─────────────────────────────────────────────────────────────

def download_prices(
    tickers: dict = TICKERS,
    start: str = TRAIN_START,
    end: str = TEST_END,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for all tickers.

    Parameters
    ----------
    tickers   : dict mapping friendly name -> Yahoo Finance ticker symbol
    start     : start date string (YYYY-MM-DD)
    end       : end date string   (YYYY-MM-DD)
    save_path : if provided, saves raw prices as CSV to this path

    Returns
    -------
    pd.DataFrame
        Columns = friendly asset names, Index = Date, values = Adj Close prices
    """
    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Rename columns from ticker symbols to friendly names
    reverse_map = {v: k for k, v in tickers.items()}
    raw = raw.rename(columns=reverse_map)

    # Drop rows where any asset has a missing value
    raw = raw.dropna()

    if save_path:
        raw.to_csv(save_path)
        print(f"Prices saved to {save_path}")

    return raw


# ── Return Computation ────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a prices DataFrame.

    Log return: r_t = log(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame of adjusted closing prices

    Returns
    -------
    pd.DataFrame of log returns (first row dropped due to differencing)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


# ── Train / Test Split ────────────────────────────────────────────────────────

def train_test_split(
    returns: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split return series into train and test sets by date.

    Parameters
    ----------
    returns    : pd.DataFrame of log returns
    train_end  : last date (inclusive) of training period
    test_start : first date (inclusive) of test period

    Returns
    -------
    tuple of (train_returns, test_returns)
    """
    train = returns.loc[:train_end]
    test  = returns.loc[test_start:]
    return train, test


# ── Summary Statistics ────────────────────────────────────────────────────────

def summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for each return series.

    Includes: mean, std, skewness, excess kurtosis, min, max, Jarque-Bera p-value.

    Parameters
    ----------
    returns : pd.DataFrame of log returns

    Returns
    -------
    pd.DataFrame with assets as columns and statistics as rows
    """
    from scipy import stats

    summary = pd.DataFrame(index=[
        "Mean", "Std Dev", "Skewness", "Excess Kurtosis", "Min", "Max", "JB p-value"
    ])

    for col in returns.columns:
        r = returns[col].dropna()
        jb_pval = stats.jarque_bera(r).pvalue
        summary[col] = [
            r.mean(),
            r.std(),
            r.skew(),
            r.kurtosis(),   # pandas kurtosis is excess kurtosis
            r.min(),
            r.max(),
            jb_pval,
        ]

    return summary.round(6)
