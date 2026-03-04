"""
var_engine.py
=============
Portfolio VaR and Expected Shortfall (ES) via Monte Carlo simulation.

Pipeline:
    1. Simulate joint uniform scenarios from a fitted copula
    2. Transform back through inverse GARCH conditional distributions
       to get simulated portfolio returns
    3. Compute VaR and ES from the simulated portfolio return distribution

VaR at level alpha: the alpha-quantile of the loss distribution
    VaR_alpha = -q_alpha(portfolio returns)

ES at level alpha: expected loss conditional on exceeding VaR
    ES_alpha = -E[r | r < -VaR_alpha]
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Core VaR / ES from Simulated Returns ─────────────────────────────────────

def compute_var_es(
    portfolio_returns: np.ndarray,
    alpha_levels: list = [0.95, 0.99],
) -> pd.DataFrame:
    """
    Compute VaR and ES at multiple confidence levels.

    Parameters
    ----------
    portfolio_returns : 1D np.ndarray of simulated or realised portfolio returns
    alpha_levels      : list of confidence levels (e.g. [0.95, 0.99])

    Returns
    -------
    pd.DataFrame with columns ['VaR', 'ES'] and confidence levels as index
    """
    results = []
    for alpha in alpha_levels:
        var = -np.quantile(portfolio_returns, 1 - alpha)
        es  = -portfolio_returns[portfolio_returns < -var].mean()
        results.append({
            "Confidence Level": f"{int(alpha*100)}%",
            "VaR":              round(var, 6),
            "ES":               round(es, 6),
        })
    return pd.DataFrame(results).set_index("Confidence Level")


# ── Full Monte Carlo Simulation Pipeline ──────────────────────────────────────

def simulate_portfolio_returns(
    copula_samples: np.ndarray,
    garch_results: dict,
    asset_names: list,
    weights: np.ndarray | None = None,
    forecast_vol: dict | None = None,
) -> np.ndarray:
    """
    Transform copula samples back to asset returns and aggregate into
    portfolio returns.

    Steps per simulation draw:
        1. u_i ~ Copula  (already done — these are copula_samples)
        2. z_i = t_nu^{-1}(u_i) or Phi^{-1}(u_i)  (inverse marginal CDF)
        3. r_i = mu_i + sigma_i * z_i               (un-standardise)
        4. r_portfolio = sum(w_i * r_i)

    Parameters
    ----------
    copula_samples : np.ndarray (n_samples, n_assets) of uniform[0,1] values
    garch_results  : dict from garch_utils.fit_all_marginals()
    asset_names    : list of asset names (must match keys of garch_results)
    weights        : portfolio weights, default = equal weight
    forecast_vol   : dict mapping asset name -> one-step-ahead conditional vol
                     (used for rolling backtest; if None, uses unconditional std)

    Returns
    -------
    np.ndarray of shape (n_samples,) — simulated portfolio returns
    """
    n_samples, n_assets = copula_samples.shape

    if weights is None:
        weights = np.ones(n_assets) / n_assets

    asset_returns = np.zeros((n_samples, n_assets))

    for i, asset in enumerate(asset_names):
        res    = garch_results[asset]
        params = res["params"]
        u_i    = copula_samples[:, i]

        # Determine degrees of freedom
        nu = params.get("nu", None)

        # Inverse CDF transform: uniform -> standardised residual
        if nu is not None:
            z_i = stats.t.ppf(u_i, df=nu)
        else:
            z_i = stats.norm.ppf(u_i)

        # Scale by conditional volatility (percentage scale from arch)
        if forecast_vol and asset in forecast_vol:
            sigma_i = forecast_vol[asset] / 100  # convert back from % scale
        else:
            sigma_i = res["conditional_vol"].mean() / 100

        # Mean (AR component — use last fitted mean as approximation)
        mu_i = res["model_result"].resid.mean() / 100

        asset_returns[:, i] = mu_i + sigma_i * z_i

    # Portfolio return = weighted sum of asset returns
    portfolio_returns = asset_returns @ weights
    return portfolio_returns


# ── One-Step Monte Carlo VaR ──────────────────────────────────────────────────

def monte_carlo_var(
    copula,
    garch_results: dict,
    asset_names: list,
    weights: np.ndarray | None = None,
    n_simulations: int = 100_000,
    alpha_levels: list = [0.95, 0.99],
    seed: int = 42,
    forecast_vol: dict | None = None,
) -> dict:
    """
    Full Monte Carlo VaR/ES pipeline using a fitted copula.

    Parameters
    ----------
    copula         : fitted GaussianCopula or StudentTCopula
    garch_results  : dict from garch_utils.fit_all_marginals()
    asset_names    : list of asset names
    weights        : portfolio weights
    n_simulations  : number of Monte Carlo draws
    alpha_levels   : VaR confidence levels
    seed           : random seed
    forecast_vol   : optional dict of one-step-ahead conditional vols

    Returns
    -------
    dict with keys:
        'var_es_table'       : pd.DataFrame of VaR/ES at each confidence level
        'portfolio_returns'  : np.ndarray of simulated returns (for plotting)
    """
    # 1. Simulate from copula
    copula_samples = copula.simulate(n_simulations, seed=seed)

    # 2. Transform to portfolio returns
    portfolio_returns = simulate_portfolio_returns(
        copula_samples, garch_results, asset_names, weights, forecast_vol
    )

    # 3. Compute VaR and ES
    var_es_table = compute_var_es(portfolio_returns, alpha_levels)

    return {
        "var_es_table":      var_es_table,
        "portfolio_returns": portfolio_returns,
    }
