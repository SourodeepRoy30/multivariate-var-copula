# Multivariate VaR with Copula Dependence Modelling

> **Research question:** Do standard univariate VaR models underestimate portfolio tail risk by ignoring cross-asset dependence structure, and can a GARCH-Copula framework produce better-calibrated risk estimates?

---

## Overview

This project implements a **GARCH-Copula framework** for multivariate portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) estimation. Standard VaR models (e.g. Historical Simulation) typically assume either independence or Gaussian dependence across assets — assumptions that break down severely during market stress when co-crash probabilities spike.

The framework applied here separates the problem into two stages:
1. **Marginal modelling:** fit a GJR-GARCH(1,1) model with Student-t innovations to each asset individually, extracting standardised residuals
2. **Dependence modelling:** fit a copula to the joint uniform marginals, comparing Gaussian (no tail dependence) vs Student-t (symmetric tail dependence) specifications

Portfolio VaR and ES are then estimated via **Monte Carlo simulation** from the fitted copula, and backtested against realised returns using Kupiec and Christoffersen tests.

---

## Assets and Data

| Asset | Ticker | Role |
|---|---|---|
| FTSE 100 | ^FTSE | UK equity market |
| GBP/USD | GBPUSD=X | Currency, risk-off hedge |
| Brent Crude | BZ=F | Commodity / global risk appetite |

- **Source:** Yahoo Finance via `yfinance`
- **Training period:** January 2010 – December 2022
- **Test period:** January 2023 – December 2024
- **Frequency:** Daily close-to-close log returns

---

## Methodology

```
Raw Prices
    │
    ▼
Log Returns (daily)
    │
    ├─── Asset 1: GJR-GARCH(1,1)-t ──► Standardised Residuals ──► PIT ──► u₁ ~ U(0,1)
    ├─── Asset 2: GJR-GARCH(1,1)-t ──► Standardised Residuals ──► PIT ──► u₂ ~ U(0,1)
    └─── Asset 3: GJR-GARCH(1,1)-t ──► Standardised Residuals ──► PIT ──► u₃ ~ U(0,1)
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
                                                            VaR (95%, 99%) + ES (97.5%)
                                                                      │
                                                            Backtest: Kupiec + Christoffersen
```

---

## Key Results

*(To be completed after model estimation)*

| Model | VaR 99% | ES 97.5% | Violations (252 days) | Kupiec p-value | Christoffersen p-value |
|---|---|---|---|---|---|
| Historical Simulation | — | — | — | — | — |
| Gaussian Copula | — | — | — | — | — |
| Student-t Copula | — | — | — | — | — |

---

## Repository Structure

```
multivariate-var-copula/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/               # Downloaded at runtime — not committed to git
│
├── notebooks/
│   ├── 01_data_and_eda.ipynb         # Data download, cleaning, EDA
│   ├── 02_garch_marginals.ipynb      # GARCH fitting, residual diagnostics
│   ├── 03_copula_fitting.ipynb       # Copula estimation, AIC/BIC comparison
│   ├── 04_var_simulation.ipynb       # Monte Carlo VaR/ES computation
│   └── 05_backtesting.ipynb          # Rolling backtest, Kupiec, Christoffersen
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Data download and preprocessing
│   ├── garch_utils.py    # GARCH fitting, PIT transform
│   ├── copula_utils.py   # Gaussian and Student-t copula
│   ├── var_engine.py     # Monte Carlo VaR/ES simulation
│   └── backtest.py       # Kupiec, Christoffersen, backtesting loop
│
└── results/
    └── figures/           # Saved plots
```

---

## Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/multivariate-var-copula.git
cd multivariate-var-copula

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in order
jupyter notebook notebooks/
```

---

## References

- McNeil, A.J., Frey, R. and Embrechts, P. (2005). *Quantitative Risk Management*. Princeton University Press.
- Engle, R.F. and Kroner, K.F. (1995). Multivariate Simultaneous Generalised ARCH. *Econometric Theory*, 11(1), 122–150.
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229–231.
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.

---

## Author

**Sourodeep Roy**  
MSc Data Science and Analytics, University of Leeds (Distinction)  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)
