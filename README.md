# Ghana Inflation Forecasting — Machine Learning Study

> **Can machine learning outperform traditional econometrics for inflation forecasting during a crisis?**
> This project answers that using 120 months of Ghanaian macroeconomic data; including the 2022 crisis where prices peaked at **54.1% YoY**.

---

## 1. Executive Summary

Ghana's inflation surged from ~7% in 2019 to a peak of **54.1% in December 2022**, driven by cedi depreciation, fuel price shocks, and fiscal pressures. This project benchmarks **13 forecasting models** on monthly CPI data from January 2015 to December 2024.

The best model, **Bayesian Ridge**, achieved an **R² of 0.919** and reduced forecast error by **72.9%** compared to the SARIMAX-GARCH baseline on the unseen 2023–2024 test period.

| Model | RMSE | MAPE | R² |
|---|---|---|---|
| 🥇 Bayesian Ridge | 3.04 | 9.1% | **0.919** |
| 🥈 Rolling XGBoost | 3.24 | **7.8%** | 0.907 |
| 🥉 Weighted Blend | 3.43 | 10.1% | 0.896 |
| Huber Regression | 4.68 | 13.7% | 0.807 |
| Elastic Net | 4.99 | 16.6% | 0.780 |
| XGBoost | 5.90 | 15.3% | 0.693 |
| Random Forest | 8.23 | 26.9% | 0.402 |
| SARIMAX-GARCH *(baseline)* | 11.18 | 29.2% | -0.103 |
| SVR | 12.41 | 22.0% | -0.358 |

---

## 2. Business Problem

The **Bank of Ghana** and economic policymakers rely on inflation forecasts to set the Monetary Policy Rate, issue forward guidance to banks and households, and plan fiscal interventions before inflation spirals out of control.

**The core challenge:** Traditional models (ARIMA, SARIMAX-GARCH) are calibrated on historical data and assume a stable relationship between variables. Ghana's 2022–2024 crisis represented a **2.3× regime shift**. The test period mean (31.6%) was more than double the training period mean (13.7%). Static models trained on pre-crisis data fail in this environment.

**Key question answered:** Which forecasting approach (traditional or ML)  best handles a sudden inflation regime shift with only 83 training observations?

---

## 3. Methodology

### Data Sources

| Variable | Description | Source |
|---|---|---|
| CPI | Year-on-year inflation % *(target)* | Ghana Statistical Service |
| PPI | Producer Price Index | Ghana Statistical Service |
| FX_interbank | USD/GHS exchange rate | Bank of Ghana |
| MPR | Monetary Policy Rate | Bank of Ghana |
| TBill91d | 91-day Treasury Bill rate | Bank of Ghana |
| IPI / EPI | Import & Export Price Indices | World Bank |
| Oil | Brent crude (USD/bbl) | World Bank |
| Gold | Gold spot price (USD/oz) | World Bank |

**Period:** Jan 2015 – Dec 2024 (120 monthly observations)
**Train:** Jan 2016 – Dec 2022 (83 obs) | **Test:** Jan 2023 – Dec 2024 (24 obs)

### Feature Engineering (26 features from 11 raw variables)

| Channel | Features |
|---|---|
| Inflation persistence | Lags L1, L2, L3 · 3-month and 6-month moving averages · rolling std |
| FX pass-through | 1-month lag · 3-month momentum · month-on-month change |
| Monetary policy | MPR at 3- and 6-month lags · policy change direction · T-bill/MPR spread |
| Energy costs | Oil at 1- and 3-month lags · YoY change |
| Trade pressures | Import and export price YoY changes |
| Interaction terms | FX × Oil (import cost pressure) · PPI × FX (cost-push) |

### Models Compared

| Category | Models |
|---|---|
| Traditional econometric | SARIMAX-GARCH |
| Regularized linear | Elastic Net · Bayesian Ridge · Huber Regression |
| Kernel method | SVR (RBF) |
| Tree ensembles | Random Forest · XGBoost · LightGBM · CatBoost · Gradient Boosting |
| Adaptive | **Rolling-Window XGBoost** — retrains on 48-month window per forecast step |
| Meta-learner | **Weighted Blend** — inverse-RMSE weights across top 4 models |

Hyperparameters tuned via time-series cross-validation (5-fold expanding window) on training set only. Final evaluation on held-out test set — no data leakage.

---

## 4. Skills

**Languages:** Python · 

**Libraries:**
`scikit-learn` · `xgboost` · `lightgbm` · `catboost` · `tensorflow/keras` · `shap` · `statsmodels` · `arch` · `pandas` · `numpy` · `matplotlib` · `seaborn`

**Techniques:**
- Machine learning: gradient boosting, regularized regression, neural networks, ensemble blending
- Statistics: SHAP feature importance, bootstrap prediction intervals, stationarity testing (ADF/KPSS), time-series cross-validation
- Econometrics: SARIMAX, GARCH volatility modelling

**Tools:** Git · Jupyter · LaTeX · Excel (openpyxl)

---

## 5. Results & Business Recommendations

### What the results show

**Bayesian Ridge wins overall (R² = 0.919)** because its probabilistic prior shrinks coefficients toward zero, preventing overfitting to the low-inflation training period when the test environment looks entirely different.

**Rolling XGBoost achieves the lowest MAPE (7.8%)** by retraining on the most recent 48 months before each forecast step. By mid-2023 it has high-inflation data in its window and adapts, while all static models remain anchored to pre-crisis patterns.

**Three key lessons:**
1. **Regularisation beats complexity with limited data.** Bayesian Ridge outperforms Random Forest, LightGBM, CatBoost and LSTM with only 83 training observations.
2. **Adaptive retraining directly solves the regime-shift problem.** Rolling-window retraining is more effective than any architectural sophistication.
3. **SHAP analysis confirms economic theory.** The top driver is lagged inflation (persistence), followed by FX level and momentum all consistent with both structuralist and monetary theories of Ghanaian inflation.

### Recommendations for the Bank of Ghana

| Use Case | Recommended Model | Why |
|---|---|---|
| Monthly point forecast | Rolling XGBoost | Lowest MAPE (7.8%); adapts in real time |
| Policy briefings | Bayesian Ridge | Interpretable; highest R² (0.919) |
| Uncertainty quantification | Bayesian Ridge + Bootstrap CI | 90% prediction intervals; 88% empirical coverage |
| Robustness check | Weighted Blend | Reduces variance when any single model fails |
| **Avoid** | SARIMAX-GARCH alone | R² = −0.10 on test set — worse than guessing the mean |

Retrain Rolling XGBoost monthly on a rolling 48-month window. Pair with Bayesian Ridge intervals for risk communication. Flag months where the two models diverge by >3 percentage points for analyst review.

---

## 6. Next Steps

- [ ] Extend to **6-month and 12-month forecast horizons**
- [ ] **Disaggregate by component** — food, core, and energy inflation separately
- [ ] **Replicate for ECOWAS economies** — Nigeria, Côte d'Ivoire, Senegal
- [ ] Implement **quantile regression forests** for full probabilistic forecast distributions
- [ ] Build a **real-time monthly pipeline** from GSS and Bank of Ghana data feeds
- [ ] Test **Temporal Fusion Transformers** as a next-generation benchmark

---

## Repository Structure

```
ghana-inflation-forecasting/
├── ghana_inflation_forecasting_v2.ipynb   # Full analysis notebook — 13 models
├── data.csv                               # Monthly data Jan 2015–Dec 2024
├── generate_results_excel.py              # Export results to formatted Excel
├── ghana_inflation_results.xlsx           # Model comparison & actual vs predicted
└── README.md
```

---

**Sharon Krasi** — Applied Statistician · Accra, Ghana
📧 krasi.sharon1@gmail.com · [linkedin.com/in/sharon-krasi](https://linkedin.com/in/sharon-krasi)
