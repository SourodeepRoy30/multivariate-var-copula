"""
copula_utils.py
===============
Copula fitting and simulation for multivariate dependence modelling.

Copulas implemented:
    1. Gaussian copula  — symmetric, no tail dependence (benchmark)
    2. Student-t copula — symmetric tail dependence (main model)

Sklar's theorem:
    Any joint distribution H(x1,...,xn) can be written as
    H(x1,...,xn) = C(F1(x1), ..., Fn(xn))
    where C is a copula and Fi are the marginal CDFs.

    This means: once we have uniform marginals u_t from GARCH-PIT,
    we fit C(u1, u2, u3) to model joint dependence independently
    of the marginal behaviour.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ── Gaussian Copula ───────────────────────────────────────────────────────────

class GaussianCopula:
    """
    Multivariate Gaussian copula.

    Parameters estimated via MLE on the uniform marginals.
    The correlation matrix R is the single parameter of this copula.

    Tail dependence: ZERO (a key limitation for financial data).
    """

    def __init__(self):
        self.R = None          # Correlation matrix
        self.n_assets = None

    def fit(self, uniforms: pd.DataFrame) -> "GaussianCopula":
        """
        Fit the Gaussian copula to uniform marginals.

        Steps:
            1. Transform uniforms back to standard normal via Phi^{-1}
            2. Estimate correlation matrix R from the normal scores
               (this is the MLE estimator for the Gaussian copula)

        Parameters
        ----------
        uniforms : pd.DataFrame with columns = assets, values in (0,1)

        Returns
        -------
        self (fitted)
        """
        self.n_assets = uniforms.shape[1]

        # Clip to avoid inf at boundaries
        u = uniforms.clip(1e-6, 1 - 1e-6)

        # Transform to normal scores
        z = stats.norm.ppf(u.values)

        # MLE for Gaussian copula = sample correlation of normal scores
        self.R = np.corrcoef(z.T)

        print("Gaussian Copula fitted.")
        print(f"  Correlation matrix:\n{pd.DataFrame(self.R, index=uniforms.columns, columns=uniforms.columns).round(4)}\n")
        return self

    def log_likelihood(self, uniforms: pd.DataFrame) -> float:
        """Compute Gaussian copula log-likelihood."""
        u = uniforms.clip(1e-6, 1 - 1e-6).values
        z = stats.norm.ppf(u)
        n = z.shape[0]

        sign, log_det_R = np.linalg.slogdet(self.R)
        R_inv = np.linalg.inv(self.R)

        # Copula log-density: -0.5 * [log|R| + z'(R^{-1} - I)z] per observation
        ll = 0.0
        for i in range(n):
            zi = z[i]
            ll += -0.5 * (log_det_R + zi @ (R_inv - np.eye(self.n_assets)) @ zi)
        return ll

    def aic_bic(self, uniforms: pd.DataFrame) -> dict:
        """Compute AIC and BIC for model comparison."""
        ll = self.log_likelihood(uniforms)
        # Free parameters: upper triangle of correlation matrix
        k = self.n_assets * (self.n_assets - 1) / 2
        n = len(uniforms)
        return {
            "log_likelihood": ll,
            "AIC": -2 * ll + 2 * k,
            "BIC": -2 * ll + k * np.log(n),
        }

    def simulate(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """
        Draw samples from the Gaussian copula.

        Steps:
            1. Draw from multivariate normal N(0, R)
            2. Transform each margin to uniform via standard normal CDF

        Parameters
        ----------
        n_samples : number of joint samples to draw
        seed      : random seed for reproducibility

        Returns
        -------
        np.ndarray of shape (n_samples, n_assets) with values in (0,1)
        """
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(
            mean=np.zeros(self.n_assets),
            cov=self.R,
            size=n_samples,
        )
        u = stats.norm.cdf(z)
        return u


# ── Student-t Copula ──────────────────────────────────────────────────────────

class StudentTCopula:
    """
    Multivariate Student-t copula.

    Parameters: correlation matrix R, degrees of freedom nu.
    Estimated via MLE — nu is optimised via grid search + refinement.

    Tail dependence: SYMMETRIC and POSITIVE.
    Lambda_L = Lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
    where rho is the pairwise correlation.

    This is more realistic than Gaussian for equity/FX/commodity portfolios.
    """

    def __init__(self):
        self.R = None
        self.nu = None
        self.n_assets = None

    def fit(self, uniforms: pd.DataFrame, nu_grid: list = None) -> "StudentTCopula":
        """
        Fit Student-t copula via two-step MLE:
            Step 1: Estimate R from Kendall's tau (rank-based, robust)
            Step 2: Grid search over nu to maximise copula log-likelihood

        Parameters
        ----------
        uniforms : pd.DataFrame of uniform marginals
        nu_grid  : list of nu values to search over (default: 2 to 30)

        Returns
        -------
        self (fitted)
        """
        self.n_assets = uniforms.shape[1]
        u = uniforms.clip(1e-6, 1 - 1e-6)

        if nu_grid is None:
            nu_grid = list(range(2, 31))

        # Step 1: Estimate R via Kendall's tau -> linear correlation
        # Pearson correlation of t-scores is a good starting point
        z_init = stats.norm.ppf(u.values)
        self.R = np.corrcoef(z_init.T)

        # Step 2: Grid search for nu
        print("Fitting Student-t Copula — searching over nu...")
        best_ll  = -np.inf
        best_nu  = None

        for nu in nu_grid:
            ll = self._log_likelihood(u.values, self.R, nu)
            if ll > best_ll:
                best_ll = ll
                best_nu = nu

        self.nu = best_nu
        print(f"  Optimal nu: {self.nu}  |  Log-likelihood: {best_ll:.2f}\n")
        return self

    @staticmethod
    def _log_likelihood(u: np.ndarray, R: np.ndarray, nu: float) -> float:
        """
        Compute Student-t copula log-likelihood.

        Copula density: c(u1,...,un) = t_{nu,R}(z1,...,zn) / prod(t_nu(zi))
        where zi = t_nu^{-1}(ui)
        """
        n, d = u.shape
        z = stats.t.ppf(u, df=nu)

        sign, log_det_R = np.linalg.slogdet(R)
        R_inv = np.linalg.inv(R)

        ll = 0.0
        # Multivariate t log density minus sum of univariate t log densities
        from scipy.special import gammaln
        log_c_mvt = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - (d / 2) * np.log(nu * np.pi)
            - 0.5 * log_det_R
        )
        log_c_uvt = d * (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
        )

        for i in range(n):
            zi = z[i]
            quad = zi @ R_inv @ zi
            ll += (
                log_c_mvt
                - ((nu + d) / 2) * np.log(1 + quad / nu)
                - log_c_uvt
                + ((nu + 1) / 2) * np.sum(np.log(1 + zi**2 / nu))
            )
        return ll

    def aic_bic(self, uniforms: pd.DataFrame) -> dict:
        """Compute AIC and BIC for model comparison."""
        u = uniforms.clip(1e-6, 1 - 1e-6).values
        ll = self._log_likelihood(u, self.R, self.nu)
        k = self.n_assets * (self.n_assets - 1) / 2 + 1  # R upper triangle + nu
        n = len(uniforms)
        return {
            "log_likelihood": ll,
            "AIC": -2 * ll + 2 * k,
            "BIC": -2 * ll + k * np.log(n),
        }

    def simulate(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """
        Draw samples from the Student-t copula.

        Steps:
            1. Draw W ~ Chi-squared(nu) / nu
            2. Draw Z ~ N(0, R)
            3. X = Z / sqrt(W)  ~  t_{nu, R}
            4. Transform to uniform via t_nu CDF

        Parameters
        ----------
        n_samples : number of joint samples
        seed      : random seed

        Returns
        -------
        np.ndarray of shape (n_samples, n_assets) with values in (0,1)
        """
        rng = np.random.default_rng(seed)

        # Multivariate t via Gaussian mixture representation
        z = rng.multivariate_normal(
            mean=np.zeros(self.n_assets),
            cov=self.R,
            size=n_samples,
        )
        w = rng.chisquare(df=self.nu, size=n_samples) / self.nu
        x = z / np.sqrt(w[:, None])

        # Transform to uniform
        u = stats.t.cdf(x, df=self.nu)
        return u


# ── Copula Comparison ─────────────────────────────────────────────────────────

def compare_copulas(
    gaussian: GaussianCopula,
    student_t: StudentTCopula,
    uniforms: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Gaussian and Student-t copulas using AIC, BIC, and log-likelihood.

    Parameters
    ----------
    gaussian  : fitted GaussianCopula
    student_t : fitted StudentTCopula
    uniforms  : pd.DataFrame of uniform marginals used for fitting

    Returns
    -------
    pd.DataFrame comparison table
    """
    g_stats = gaussian.aic_bic(uniforms)
    t_stats = student_t.aic_bic(uniforms)

    comparison = pd.DataFrame(
        {
            "Gaussian Copula":   g_stats,
            "Student-t Copula":  t_stats,
        }
    )
    return comparison.round(3)
