"""
garch_utils.py
==============
Univariate GARCH marginal model fitting for each asset.

Pipeline per asset:
    1. Fit AR(1)-GJR-GARCH(1,1) with Student-t innovations
    2. Extract standardised residuals
    3. Apply Probability Integral Transform (PIT) → uniform marginals
       (these uniform marginals are the inputs to copula fitting)

Model choice rationale:
    - GJR-GARCH captures the leverage effect (asymmetric volatility)
    - Student-t innovations handle fat tails in financial returns
    - AR(1) mean equation handles mild autocorrelation in returns
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t
from scipy.stats import norm


# ── GARCH Fitting ─────────────────────────────────────────────────────────────

def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "studentst",
) -> dict:
    """
    Fit an AR(1)-GJR-GARCH(p,o,q) model to a univariate return series.

    Parameters
    ----------
    returns : pd.Series of log returns (should be scaled by 100 for arch library)
    p       : GARCH lag order
    o       : asymmetry (GJR) lag order
    q       : ARCH lag order
    dist    : innovation distribution ('studentst' or 'normal')

    Returns
    -------
    dict with keys:
        'model_result'      : fitted arch ModelResult object
        'std_residuals'     : standardised residuals (epsilon_t)
        'conditional_vol'   : conditional volatility sigma_t
        'params'            : fitted parameter Series
    """
    # arch library works better with returns scaled to percentage
    r_scaled = returns * 100

    model = arch_model(
        r_scaled,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist,
    )

    result = model.fit(disp="off", show_warning=False)

    std_resid      = result.std_resid
    conditional_vol = result.conditional_volatility

    return {
        "model_result":    result,
        "std_residuals":   std_resid,
        "conditional_vol": conditional_vol,
        "params":          result.params,
    }


def fit_all_marginals(
    returns: pd.DataFrame,
    dist: str = "studentst",
) -> dict:
    """
    Fit GJR-GARCH models to all assets in the returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame where each column is an asset's log return series
    dist    : innovation distribution for all models

    Returns
    -------
    dict mapping asset name -> fit results dict (from fit_gjr_garch)
    """
    results = {}
    for col in returns.columns:
        print(f"Fitting GJR-GARCH for {col}...")
        results[col] = fit_gjr_garch(returns[col], dist=dist)
        params = results[col]["params"]
        print(f"  AIC: {results[col]['model_result'].aic:.2f}  |  "
              f"BIC: {results[col]['model_result'].bic:.2f}\n")
    return results


# ── Probability Integral Transform (PIT) ─────────────────────────────────────

def pit_transform(
    std_residuals: pd.Series,
    dist: str = "studentst",
    nu: float | None = None,
) -> pd.Series:
    """
    Transform standardised residuals to uniform[0,1] via the PIT.

    If dist='studentst', uses the fitted Student-t CDF with nu degrees of freedom.
    If dist='normal', uses the standard normal CDF.

    The PIT is: u_t = F(z_t) where F is the marginal CDF.
    If the GARCH model is correctly specified, u_t ~ Uniform(0,1).

    Parameters
    ----------
    std_residuals : pd.Series of standardised GARCH residuals
    dist          : 'studentst' or 'normal'
    nu            : degrees of freedom (required if dist='studentst')

    Returns
    -------
    pd.Series of uniform[0,1] values (u_t)
    """
    z = std_residuals.dropna()

    if dist == "studentst":
        if nu is None:
            raise ValueError("Provide degrees of freedom (nu) for Student-t PIT.")
        uniforms = student_t.cdf(z, df=nu)
    else:
        uniforms = norm.cdf(z)

    return pd.Series(uniforms, index=z.index, name=std_residuals.name)


def extract_uniforms(garch_results: dict) -> pd.DataFrame:
    """
    Extract uniform marginals from all fitted GARCH models.

    For each asset, retrieves the nu (degrees of freedom) parameter from
    the fitted model and applies the PIT.

    Parameters
    ----------
    garch_results : dict from fit_all_marginals()

    Returns
    -------
    pd.DataFrame where each column is an asset's uniform marginal u_t
    """
    uniforms = {}
    for asset, res in garch_results.items():
        params = res["params"]
        # arch library names the nu parameter 'nu'
        if "nu" in params.index:
            nu = params["nu"]
            u = pit_transform(res["std_residuals"], dist="studentst", nu=nu)
        else:
            u = pit_transform(res["std_residuals"], dist="normal")
        uniforms[asset] = u

    df_uniforms = pd.DataFrame(uniforms).dropna()
    return df_uniforms


# ── Diagnostics ───────────────────────────────────────────────────────────────

def garch_diagnostics(model_result) -> dict:
    """
    Run basic diagnostics on a fitted GARCH model.

    Checks:
        - Ljung-Box test on standardised residuals (serial correlation)
        - Ljung-Box test on squared standardised residuals (remaining ARCH effects)

    Parameters
    ----------
    model_result : fitted arch ModelResult object

    Returns
    -------
    dict with test statistics and p-values
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    std_resid = model_result.std_resid.dropna()

    lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
    lb_sq    = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)

    return {
        "LB_resid_stat":   lb_resid["lb_stat"].values[0],
        "LB_resid_pval":   lb_resid["lb_pvalue"].values[0],
        "LB_sq_resid_stat": lb_sq["lb_stat"].values[0],
        "LB_sq_resid_pval": lb_sq["lb_pvalue"].values[0],
    }
