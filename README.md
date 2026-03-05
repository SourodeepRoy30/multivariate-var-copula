# Multivariate VaR with Copula Dependence Modelling

> **Research question:** Do standard univariate VaR models underestimate portfolio tail risk by ignoring cross-asset dependence structure, and can a GARCH-Copula framework produce better-calibrated risk estimates?

---

## Overview

This project implements a **GARCH-Copula framework** for multivariate portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) estimation across a three-asset portfolio of FTSE 100, GBP/USD, and Brent Crude Oil.

Standard VaR models typically assume either independence or Gaussian dependence across assets — assumptions that break down severely during market stress when co-crash probabilities spike. The 2008 financial crisis famously exposed the Gaussian copula's inability to capture joint tail dependence, contributing to widespread underestimation of systemic risk.

This project separates the problem into two stages following Sklar's theorem:

1. **Marginal modelling:** fit an AR(1)-GJR-GARCH(1,1) model with Student-t innovations to each asset individually, capturing volatility clustering, leverage effects, and fat tails
2. **Dependence modelling:** fit a copula to the joint uniform marginals, comparing Gaussian (zero tail dependence) vs Student-t (symmetric tail dependence) specifications

Portfolio VaR and ES are estimated via **Monte Carlo simulation** (100,000 scenarios) from each fitted copula and backtested against realised returns using Kupiec and Christoffersen tests.

---

## Assets and Data

| Asset | Source | Role in Portfolio |
|-------|--------|------------------|
| FTSE 100 | Investing.com | UK equity market |
| GBP/USD | Investing.com | Currency — risk-off indicator |
| Brent Crude | Investing.com | Commodity — global risk appetite |

- **Training period:** January 2010 – December 2022 (3,280 observations)
- **Test period:** January 2023 – December 2024 (505 observations)
- **Frequency:** Daily close-to-close log returns
- **Portfolio weights:** Equal-weighted (1/3 each)

---

## Methodology

```
Raw Prices (Investing.com)
        │
        ▼
Log Returns r_t = log(P_t / P_{t-1})
        │
        ├── FTSE100: AR(1)-GJR-GARCH(1,1)-t ──► z_t ──► PIT ──► u₁ ~ U(0,1)
        ├── GBPUSD:  AR(1)-GJR-GARCH(1,1)-t ──► z_t ──► PIT ──► u₂ ~ U(0,1)
        └── Brent:   AR(1)-GJR-GARCH(1,1)-t ──► z_t ──► PIT ──► u₃ ~ U(0,1)
                                                                      │
                                                      ┌───────────────┤
                                                      │               │
                                            Gaussian Copula   Student-t Copula
                                                      │               │
                                                      └───────┬───────┘
                                                              │
                                                    Monte Carlo Simulation
                                                        (100,000 scenarios)
                                                              │
                                                  Portfolio Return Distribution
                                                              │
                                              VaR (95%, 99%) + ES (95%, 99%)
                                                              │
                                              Backtest: Kupiec + Christoffersen
```

---

## EDA Highlights

All three return series exhibit significant departure from normality — a key motivation for fat-tailed modelling:

| Statistic | FTSE 100 | GBP/USD | Brent Crude |
|-----------|---------|---------|-------------|
| Mean | 0.000093 | -0.000087 | 0.000021 |
| Std Dev | 0.010334 | 0.005821 | 0.022990 |
| Skewness | -0.684 | -0.987 | -0.971 |
| Excess Kurtosis | 9.57 | 15.45 | 18.08 |
| JB p-value | 0.000 | 0.000 | 0.000 |
| ADF p-value | 0.000 | 0.000 | 0.000 |
| ARCH LM p-value | 0.000 | 0.000 | 0.000 |

- All three series reject normality (Jarque-Bera), confirm stationarity (ADF), and exhibit strong ARCH effects — justifying the GJR-GARCH specification
- Rolling correlations show substantial time variation and regime-dependent behaviour — particularly during the COVID crash (2020) and energy crisis (2022)
- Corner concentration analysis confirms joint crash frequency exceeds Gaussian predictions by 1.4–2.3×

---

## GARCH Marginal Model Results

AR(1)-GJR-GARCH(1,1) with Student-t innovations fitted to each asset:

| Parameter | FTSE 100 | GBP/USD | Brent Crude |
|-----------|---------|---------|-------------|
| $\gamma$ (leverage) | 0.2355*** | 0.0402*** | 0.0660*** |
| $\beta$ (persistence) | 0.846*** | 0.943*** | 0.914*** |
| $\nu$ (tail thickness) | 6.75 | 7.50 | 5.56 |
| Ljung-Box resid p | 0.620 | 0.929 | 0.471 |
| Ljung-Box sq. resid p | 0.810 | 0.265 | 0.671 |

*Significance: *** p < 0.001*

- Leverage effect ($\gamma$) is significant for all three assets — negative shocks increase volatility more than positive shocks
- High persistence ($\beta$ > 0.84) confirms volatility clustering
- All models pass Ljung-Box diagnostics — no remaining serial correlation or ARCH effects in residuals

---

## Copula Fitting Results

### Model Comparison

| Metric | Gaussian Copula | Student-t Copula |
|--------|----------------|-----------------|
| Log-Likelihood | 241.66 | 318.49 |
| AIC | -477.31 | -628.97 |
| BIC | -459.02 | -604.59 |
| $\Delta$ AIC | — | **151.66 better** |
| Degrees of freedom $\nu$ | — | 5 |

The Student-t copula is decisively preferred — an AIC improvement of 151.66 from a single additional parameter.

### Tail Dependence Coefficients

| Pair | Correlation $\rho$ | Student-t $\lambda$ | Gaussian $\lambda$ |
|------|-------------------|--------------------|--------------------|
| FTSE 100 / GBP-USD | 0.018 | 0.053 | 0.000 |
| FTSE 100 / Brent   | 0.322 | 0.130 | 0.000 |
| GBP-USD / Brent    | 0.188 | 0.089 | 0.000 |

The Gaussian copula assigns zero probability to simultaneous extreme moves regardless of correlation. The Student-t copula correctly identifies a 13% conditional probability of FTSE/Brent co-crashing — a scenario that must be reserved against.

---

## VaR and ES Results

Monte Carlo simulation — 100,000 scenarios, equal-weighted portfolio:

| Model | VaR 95% | ES 95% | VaR 99% | ES 99% |
|-------|---------|--------|---------|--------|
| Historical Simulation | 0.0152 | 0.0239 | 0.0274 | 0.0418 |
| Gaussian Copula | 0.0178 | 0.0243 | 0.0281 | 0.0354 |
| Student-t Copula | 0.0177 | 0.0248 | 0.0290 | 0.0373 |

**Student-t vs Gaussian at 99%: VaR +3.02%, ES +5.38%**

A risk manager using the Gaussian copula would reserve 3.02% less capital at the 99% VaR level — a systematic underestimation that scales directly with portfolio size.

---

## Backtesting Results (Test Period 2023–2024)

| Model | Confidence | Violations | Expected | Violation Rate | Kupiec Reject | Christ. Reject |
|-------|-----------|-----------|----------|----------------|---------------|----------------|
| Historical Simulation | 95% | 5  | 25 | 0.0099 | Yes | Yes |
| Historical Simulation | 99% | 0  | 5  | 0.0000 | Yes | Yes |
| Gaussian Copula | 95% | 9  | 25 | 0.0178 | Yes | Yes |
| Gaussian Copula | 99% | 1  | 5  | 0.0020 | Yes | No  |
| Student-t Copula | 95% | 9  | 25 | 0.0178 | Yes | No  |
| Student-t Copula | 99% | 1  | 5  | 0.0020 | Yes | No  |

- All models are over-conservative — the 2023–2024 test period was significantly calmer than the crisis-heavy training period (2010–2022)
- Both copula models produce identical violation counts — the test period lacked the systemic tail events needed to empirically differentiate them
- The Student-t copula passes the Christoffersen independence test at both confidence levels — violations are isolated, not clustered
- Historical Simulation severely over-embeds past crises into forward-looking forecasts — failing both tests at both levels

---

## Repository Structure

```
multivariate-var-copula/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── raw/                        # Downloaded at runtime — not committed
│
├── notebooks/
│   ├── 01_data_and_eda.ipynb       # Data loading, cleaning, EDA
│   ├── 02_garch_marginals.ipynb    # GARCH fitting, diagnostics, PIT
│   ├── 03_copula_fitting.ipynb     # Copula estimation, comparison
│   ├── 04_var_simulation.ipynb     # Monte Carlo VaR/ES simulation
│   └── 05_backtesting.ipynb        # Kupiec, Christoffersen, violation plots
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data download and preprocessing
│   ├── garch_utils.py              # GJR-GARCH fitting, PIT transform
│   ├── copula_utils.py             # Gaussian and Student-t copula
│   ├── var_engine.py               # Monte Carlo VaR/ES simulation
│   └── backtest.py                 # Kupiec, Christoffersen, backtesting loop
│
└── results/
    └── figures/                    # All saved plots
```

---

## Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/SourodeepRoy30/multivariate-var-copula.git
cd multivariate-var-copula

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data
# Download FTSE 100, GBP/USD, and Brent Crude historical data from
# Investing.com and place in data/raw/ as:
#   FTSE 100 Historical Data.csv
#   GBP_USD Historical Data.csv
#   Brent Oil Futures Historical Data.csv

# 5. Run notebooks in order
jupyter notebook notebooks/
```

---

## Key Visualisations

| Figure | Description |
|--------|-------------|
| `01_return_series.png` | Daily log returns with crisis period annotations |
| `02_rolling_correlations.png` | 60-day rolling pairwise correlations |
| `03_return_distributions.png` | Return histograms vs normal distribution |
| `04_acf_plots.png` | ACF of returns and squared returns |
| `05_qq_plots.png` | Q-Q plots confirming fat tails |
| `06_correlation_heatmap.png` | Static pairwise correlation matrix |
| `07_conditional_volatility.png` | GJR-GARCH conditional volatility |
| `08_uniform_marginals.png` | PIT uniform marginals verification |
| `09_uniform_scatter.png` | Joint dependence scatter plots |
| `10_copula_comparison.png` | Empirical vs simulated copula samples |
| `11_var_distributions.png` | Monte Carlo portfolio return distributions |
| `12_tail_zoom.png` | Left tail comparison — VaR and ES |
| `13_var_violations.png` | VaR violation plot — test period |

---

## Limitations and Future Work

- **Static VaR forecasts** — a full rolling re-estimation window would produce time-varying VaR forecasts better reflecting changing market conditions
- **Symmetric tail dependence** — the Student-t copula assumes $\lambda_L = \lambda_U$. A Clayton copula allowing asymmetric lower tail dependence may be more realistic for equity/commodity portfolios
- **Three-asset portfolio** — extending to a larger, more diversified portfolio would provide a more rigorous test of the framework
- **Short test period** — 505 observations is insufficient for robust backtesting of 99% VaR. A longer window including a systemic crisis would be the definitive evaluation

---

## References

- McNeil, A.J., Frey, R. and Embrechts, P. (2005). *Quantitative Risk Management*. Princeton University Press.
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229–231.
- Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987–1007.
- Glosten, L.R., Jagannathan, R. and Runkle, D.E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779–1801.
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841–862.
- Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). Evaluating Density Forecasts with Applications to Financial Risk Management. *International Economic Review*, 39(4), 863–883.

---

## Author

**Sourodeep Roy**  
MSc Data Science and Analytics, University of Leeds (Distinction)  
[LinkedIn](https://linkedin.com/in/sourodeep-roy) | [GitHub](https://github.com/SourodeepRoy30)
