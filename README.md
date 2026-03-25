# Ghana Inflation Forecasting — Machine Learning Study

Comparing **13 forecasting models** on Ghana's monthly inflation data (2015–2024), including a challenging test period covering the 2022 inflation crisis (peak: **54.1% YoY**) and subsequent disinflation through 2024.

---

## Results

| Rank | Model | RMSE | MAPE | R² |
|------|-------|------|------|----|
| 🥇 | Bayesian Ridge | 3.04 | 9.1% | **0.919** |
| 🥈 | Rolling XGBoost | 3.24 | **7.8%** | 0.907 |
| 🥉 | Weighted Blend | 3.43 | 10.1% | 0.896 |
| 4 | Huber Regression | 4.68 | 13.7% | 0.807 |
| 5 | Elastic Net | 4.99 | 16.6% | 0.780 |
| 6 | XGBoost | 5.90 | 15.3% | 0.693 |
| — | SARIMAX-GARCH (baseline) | 11.18 | 29.2% | -0.103 |

**Top 3 models reduce RMSE by 69–73% vs the traditional SARIMAX-GARCH baseline.**

---

## Key findings

**Bayesian Ridge wins** because its probabilistic prior shrinks coefficients toward zero  exactly right when the training period (mean inflation: 13.7%) looks very different from the test period (mean: 31.6%). It doesn't overfit to the old regime.

**Rolling XGBoost** solves the regime-shift problem directly by retraining on the most recent 48 months before each prediction. By mid-2023 it has high-inflation data in its window and adapts, while static models stay anchored to pre-crisis patterns.

**Simple models beat complex ensembles** with only 83 training observations. Stacking and multivariate deep learning overfit badly. Regularization and adaptive retraining matter more than architectural complexity when data is scarce.

---

## Models compared

| Category | Models |
|---|---|
| Traditional econometric | SARIMAX-GARCH |
| Regularized regression | Elastic Net, Bayesian Ridge, Huber Regression |
| Kernel method | SVR (RBF) |
| Tree ensembles | Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting |
| Adaptive | Rolling-Window XGBoost |
| Meta-learner | Weighted Blend (inverse-RMSE weights) |

---

## Data

**Period:** January 2015 – December 2024 (120 monthly observations)  
**Train:** January 2016 – December 2022 (83 obs) | **Test:** January 2023 – December 2024 (24 obs)

| Variable | Description | Source |
|---|---|---|
| Inflation (YoY %) | Target variable | Ghana Statistical Service |
| CPI, PPI | Consumer and Producer Price Indices | Ghana Statistical Service |
| FX_interbank | USD/GHS exchange rate | Bank of Ghana |
| MPR, TBill91d | Monetary Policy Rate, 91-day T-bill | Bank of Ghana |
| IPI, EPI | Import and Export Price Indices | World Bank |
| Oil, Gold | Brent crude (USD/bbl), Gold spot (USD/oz) | World Bank |

26 engineered features capture inflation persistence, FX pass-through, monetary policy lags, energy costs, trade prices, and interaction terms.

---

## Getting started

```bash
git clone https://github.com/YOUR-USERNAME/ghana-inflation-forecasting
cd ghana-inflation-forecasting
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost shap statsmodels arch-py tensorflow
jupyter notebook ghana_inflation_forecasting_v2.ipynb
```

Or just run the install cell at the top of the notebook, then **Kernel → Restart & Run All**.

---

## Author

**Sharon Krasi** — Applied Statistician  
📍 Accra, Ghana | 📧 krasi.sharon1@gmail.com | 🔗 [linkedin.com/in/sharon-krasi](https://linkedin.com/in/sharon-krasi)
