"""
backtest.py
===========
Rolling window VaR backtesting with statistical coverage tests.

Tests implemented:
    1. Kupiec POF (Proportion of Failures) test
       H0: E[violations] = 1 - alpha  (correct unconditional coverage)

    2. Christoffersen Interval Forecast test
       H0: violations are i.i.d. Bernoulli (correct coverage + independence)

    3. Historical Simulation VaR (benchmark to compare against copula VaR)

Rolling window approach:
    For each day t in the test period:
        - Train on [t - window_size, t-1]
        - Produce one-day-ahead VaR forecast
        - Record whether actual return at t is a violation
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Kupiec POF Test ───────────────────────────────────────────────────────────

def kupiec_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) likelihood ratio test.

    H0: p = 1 - alpha (correct unconditional coverage)
    H1: p != 1 - alpha

    LR = -2 * log[ p0^x * (1-p0)^(n-x) / p_hat^x * (1-p_hat)^(n-x) ]
    LR ~ Chi-squared(1) under H0

    Parameters
    ----------
    violations : binary array, 1 where actual loss exceeds VaR, 0 otherwise
    alpha      : VaR confidence level (e.g. 0.99)

    Returns
    -------
    dict with 'n_violations', 'expected_violations', 'LR_stat', 'p_value', 'reject_H0'
    """
    n   = len(violations)
    x   = violations.sum()
    p0  = 1 - alpha
    p_hat = x / n

    if p_hat == 0:
        lr_stat = 2 * n * np.log(1 / (1 - p0))
    elif p_hat == 1:
        lr_stat = 2 * n * np.log(1 / p0)
    else:
        lr_stat = -2 * (
            x * np.log(p0 / p_hat) + (n - x) * np.log((1 - p0) / (1 - p_hat))
        )

    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_obs":                n,
        "n_violations":         int(x),
        "expected_violations":  round(n * p0, 2),
        "violation_rate":       round(p_hat, 4),
        "LR_stat":              round(lr_stat, 4),
        "p_value":              round(p_value, 4),
        "reject_H0 (5%)":       p_value < 0.05,
    }


# ── Christoffersen Test ───────────────────────────────────────────────────────

def christoffersen_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Christoffersen Interval Forecast test.

    Tests jointly:
        1. Correct unconditional coverage (Kupiec)
        2. Independence of violations (no clustering)

    Total LR = LR_uc + LR_ind ~ Chi-squared(2) under H0

    Parameters
    ----------
    violations : binary array of VaR exceedances
    alpha      : VaR confidence level

    Returns
    -------
    dict with LR statistics and p-values for independence and joint test
    """
    n = len(violations)
    v = violations.astype(int)

    # Transition counts
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi   = (n01 + n11) / n

    # Independence LR
    def safe_log(x):
        return np.log(x) if x > 0 else 0

    lr_ind = -2 * (
        n00 * safe_log(1 - pi) + n01 * safe_log(pi)
        + n10 * safe_log(1 - pi) + n11 * safe_log(pi)
        - n00 * safe_log(1 - pi01) - n01 * safe_log(pi01 + 1e-10)
        - n10 * safe_log(1 - pi11 + 1e-10) - n11 * safe_log(pi11 + 1e-10)
    )

    # Unconditional coverage LR (Kupiec component)
    p0    = 1 - alpha
    x     = v.sum()
    p_hat = x / n
    lr_uc = -2 * (
        x * safe_log(p0 / (p_hat + 1e-10))
        + (n - x) * safe_log((1 - p0) / (1 - p_hat + 1e-10))
    )

    # Joint LR
    lr_joint = lr_uc + lr_ind

    p_ind   = 1 - stats.chi2.cdf(lr_ind, df=1)
    p_joint = 1 - stats.chi2.cdf(lr_joint, df=2)

    return {
        "LR_independence":          round(lr_ind, 4),
        "p_value_independence":     round(p_ind, 4),
        "LR_joint":                 round(lr_joint, 4),
        "p_value_joint":            round(p_joint, 4),
        "reject_independence (5%)": p_ind < 0.05,
        "reject_joint (5%)":        p_joint < 0.05,
    }


# ── Historical Simulation Benchmark ──────────────────────────────────────────

def historical_simulation_var(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.99,
) -> pd.Series:
    """
    Rolling Historical Simulation VaR (benchmark model).

    For each day t, VaR is the alpha-quantile of the past `window` returns.
    No parametric assumptions — simplest possible VaR model.

    Parameters
    ----------
    returns : pd.Series of actual portfolio returns
    window  : rolling window size in days
    alpha   : confidence level

    Returns
    -------
    pd.Series of rolling VaR estimates aligned with returns index
    """
    var_hs = returns.rolling(window).quantile(1 - alpha).abs()
    return var_hs


# ── Backtest Summary ──────────────────────────────────────────────────────────

def backtest_summary(
    actual_returns: pd.Series,
    var_forecasts: pd.Series,
    alpha: float,
    model_name: str = "Model",
) -> dict:
    """
    Run full backtest for a single VaR model.

    Parameters
    ----------
    actual_returns : pd.Series of actual portfolio returns (test period)
    var_forecasts  : pd.Series of VaR forecasts (must be positive values)
    alpha          : confidence level used for VaR
    model_name     : label for display

    Returns
    -------
    dict with violations array, Kupiec results, Christoffersen results
    """
    # Align on common dates
    common = actual_returns.index.intersection(var_forecasts.index)
    r      = actual_returns.loc[common]
    v_hat  = var_forecasts.loc[common]

    # Violation indicator: 1 if actual loss exceeds VaR forecast
    violations = (r < -v_hat).astype(int).values

    kupiec = kupiec_test(violations, alpha)
    christ = christoffersen_test(violations, alpha)

    return {
        "model":           model_name,
        "alpha":           alpha,
        "violations":      violations,
        "kupiec":          kupiec,
        "christoffersen":  christ,
    }


def format_backtest_results(results_list: list) -> pd.DataFrame:
    """
    Format a list of backtest result dicts into a clean summary DataFrame.

    Parameters
    ----------
    results_list : list of dicts from backtest_summary()

    Returns
    -------
    pd.DataFrame suitable for printing / saving to README
    """
    rows = []
    for r in results_list:
        k = r["kupiec"]
        c = r["christoffersen"]
        rows.append({
            "Model":             r["model"],
            "Confidence":        f"{int(r['alpha']*100)}%",
            "Violations":        k["n_violations"],
            "Expected":          k["expected_violations"],
            "Violation Rate":    k["violation_rate"],
            "Kupiec p-value":    k["p_value"],
            "Kupiec Reject":     k["reject_H0 (5%)"],
            "Christ. p-value":   c["p_value_joint"],
            "Christ. Reject":    c["reject_joint (5%)"],
        })
    return pd.DataFrame(rows)
