"""Builds notebooks/financial_time_series_analysis.ipynb (run once; then execute the notebook)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Forecasting next-day volatility of Commercial Bank of Ceylon (COMB.N0000)
## Traditional time-series models vs machine learning — CSE daily data, Jan 2024 – Dec 2025

**Research question.** *How accurately can the next trading day's price volatility of COMB voting shares be forecast
using traditional time-series models (ARIMA on log-volatility, GARCH) compared with machine-learning models
(Ridge regression, Random Forest, Gradient Boosting), relative to simple baselines, when evaluated one day ahead on an
untouched final period (Aug–Dec 2025) — and does the answer change between calm and turbulent market conditions?*

This notebook runs the complete analysis step by step. Every function lives in `src/` so that `python main.py`
produces exactly the same numbers. Run cells from top to bottom (Kernel → Restart & Run All).""")

code("""import sys, warnings, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

from src import config, data_processing as dp, feature_engineering as fe
from src import exploratory_analysis as eda, diagnostics as dg, additional_analysis as aa
from src.pipeline import run_models
show = lambda p: display(Image(filename=str(p), width=950))""")

md("""## 1. Inspect the raw dataset
**What we did and why.** Before choosing any method we looked at what the file actually contains. It turned out to mix
*three different securities*: voting shares (N), non-voting shares (X) and rights entitlements (R) that traded only for six days
during the 2024 rights issue. Treating them as one series would be a serious error.""")
code("""raw = dp.load_raw()
print("Raw shape:", raw.shape)
display(raw.head())
display(dp.raw_overview(raw))""")

md("""### Data-quality checks (voting share)
**Key findings.** No duplicates, chronological order, no impossible prices. Four LOW and two OPEN values are missing (2024).
**The OPEN column equals CLOSE on 100% of days in 2025** — it is not a real opening price in this export, so we never use it.
Gaps of 2–5 calendar days are weekends and CSE holidays, not missing data.""")
code("""display(dp.quality_report(raw, "N"))""")

md("""## 2. Cleaning, corporate actions and the target
**What we did and why.**
* Missing LOW prices were filled with min(open, close) of the *same* day (no future information).
* Unadjusted closing prices drop mechanically on ex-dividend and ex-rights days. We corrected the close-to-close returns on
  those three dates (dividend added back; rights issue via the theoretical ex-rights price).
* **Target:** the Parkinson range-based daily volatility, σ = 100·|ln(High/Low)| / √(4 ln 2). It uses the whole day's
  trading range, so it measures daily risk much more precisely than one squared return, and it is unaffected by dividends.""")
code("""g = dp.build_dataset("N")
g_x = dp.build_dataset("X")
display(dp.corporate_action_table(g, "N"))
display(g[["high","low","close","volume","ret","park_vol"]].head())""")

md("""## 3. Exploratory data analysis""")
code("""display(eda.descriptive_stats(g))
display(eda.period_stats(g))""")
code("""show(eda.fig_price_history(g, g_x))""")
md("""**Interpretation.** The voting share rose from about Rs 97 to Rs 201 (unadjusted) — a strong bull market for Sri Lankan banks.
The red lines mark the ex-dividend (XD) and ex-rights (XR) days that we corrected for. A rising price level is non-stationary,
which is one reason we do not forecast the price itself.""")
code("""show(eda.fig_returns(g)); show(eda.fig_return_distribution(g))""")
md("""**Interpretation.** Returns show bursts of large moves (the rights-issue announcement in May 2024, the US-tariff sell-off of April 2025),
fat tails (Jarque–Bera rejects normality) and a rolling standard deviation that changes over time — i.e. *risk itself varies*.
This motivates forecasting volatility rather than returns.""")
code("""show(eda.fig_target(g)); show(eda.fig_target_distribution(g))
print("Kruskal-Wallis weekday test (training):", eda.weekday_test(g))""")
md("""**Interpretation.** Volatility is strongly right-skewed; its logarithm is close to symmetric, so ARIMA and ML models are fitted to
ln(volatility). No weekday pattern was found (Kruskal–Wallis p ≈ 0.88), so a seasonal model (SARIMA) is not justified.
The extreme value on 1 Dec 2025 (intraday low Rs 170 vs close Rs 200.25, the first trading day after Cyclone Ditwah) is kept, not deleted,
and analysed separately later.""")

md("""## 4. Statistical tests (training sample only)
Decisions about differencing and model structure use only training data, so test information cannot leak into model design.""")
code("""display(eda.stationarity_tests(g)); display(eda.dependence_tests(g)); show(eda.fig_acf(g))""")
md("""**Interpretation.**
* log price: non-stationary (ADF cannot reject a unit root; KPSS rejects stationarity) → price levels are not modelled.
* returns and log-volatility: stationary → ARIMA with **d = 0** on log-volatility.
* log-volatility has very strong autocorrelation (Ljung–Box p < 0.001) → volatility is forecastable.
* Squared returns show *no* significant autocorrelation and the ARCH-LM test is not significant. Volatility clustering exists
  (see log-vol), but squared daily returns are too noisy to reveal it. This is an early warning that GARCH, which learns only from
  squared returns, may be at a disadvantage.
* Returns themselves are autocorrelated (Ljung–Box p < 0.001) — common in thinly traded frontier markets — but direction forecasting
  is not our objective.""")

md("""## 5. Features and the chronological split
Each row is a forecast for day t+1 made at the close of day t; every feature is shifted so it only uses data up to day t.""")
code("""d = fe.build_features(g)
train, val, test = fe.split(d)
display(fe.split_table(train, val, test))
display(pd.DataFrame({"feature": fe.FEATURES, "description": [fe.FEATURE_DESCRIPTIONS[f] for f in fe.FEATURES]}))
show(eda.fig_correlation(train))""")

md("""## 6. Fit all models
`run_models` performs, in order: ARIMA order by AIC on training data → GARCH variant by validation QLIKE →
ML hyper-parameters by validation QLIKE → re-estimation on training + validation → one forecast per test day with fixed parameters.""")
code("""res = run_models("N")
pred = res["pred"]
display(res["details"]["arima"]["selection"].head(6))
display(dg.arima_params_table(res))
display(res["details"]["garch"]["selection"])
display(dg.garch_params_table(res))
for k, v in res["details"]["ml"].items():
    print(k, "-> best:", v["best"]); display(v["tuning"].head(5))""")
md("""**Interpretation.** ARIMA(1,0,1) on log-volatility has AR = 0.77 and MA = −0.44: volatility shocks persist but partly fade
within days. The GARCH(1,1)-t parameters are imprecise (β only marginally significant, heavy tails ν ≈ 3.8), consistent with the weak
ARCH evidence in squared returns. Ridge selected the smallest penalty (α = 0.01), i.e. effectively ordinary least squares;
the trees chose shallow-to-moderate depth.""")

md("""## 7. Results: validation and test comparison
Metrics: **QLIKE** (primary; robust to noisy volatility proxies, penalises under-predicting risk), RMSE and MAE in percentage points
of daily volatility, Mincer–Zarnowitz R².""")
code("""comp = dg.comparison_table(res)
display(comp[["approach","train QLIKE","val QLIKE","test QLIKE","val RMSE","test RMSE","test MAE","test MZ R2","val rank (QLIKE)","test rank (QLIKE)"]])
display(dg.improvement_table(comp))""")
code("""print("Diebold-Mariano vs 22-day historical mean (test):"); display(dg.dm_table(res, "Historical mean (22-day)"))
print("Diebold-Mariano vs Ridge (best on validation):"); display(dg.dm_table(res, comp["val QLIKE"].idxmin()))""")
code("""show(dg.fig_full_forecast(res, comp["val QLIKE"].idxmin()))
show(dg.fig_forecasts_test(res, "Traditional", "15a")); show(dg.fig_forecasts_test(res, "Machine learning", "15b"))""")

md("""## 8. Diagnostics""")
code("""display(dg.residual_tests(res))
show(dg.fig_arima_residuals(res)); show(dg.fig_garch_diagnostics(res)); show(dg.fig_ml_diagnostics(res))""")

md("""## 9. Additional investigation: volatility regimes, stability and shocks
Regime of each forecast day = high if the trailing 22-day mean volatility (known at the forecast origin) exceeds its training median.""")
code("""reg, thr = aa.regime_table(res); reg_x, _ = aa.regime_table(res, exclude=["2025-12-01"])
print(f"Regime threshold: {thr:.3f}% daily volatility")
display(reg.pivot(index="model", columns="regime", values="QLIKE"))
display(reg_x.pivot(index="model", columns="regime", values="QLIKE"))
show(aa.fig_regimes(res, reg, reg_x))
roll = aa.rolling_loss(res); show(aa.fig_rolling_loss(res, roll))
display(aa.event_windows(res).pivot(index="model", columns="event", values="QLIKE"))
display(aa.outlier_sensitivity(res))
display(dg.dm_table(res, "Historical mean (22-day)", exclude=["2025-12-01"]))""")
code("""imp = aa.permutation_importance(res)
display(imp.pivot(index="feature", columns="model", values="mean QLIKE increase").sort_values("Ridge regression", ascending=False))
show(aa.fig_importance(res, imp))
display(aa.ridge_coefficients(res))""")

md("""## 10. Robustness: same pipeline on the non-voting share (COMB.X0000)""")
code("""res_x = run_models("X")
display(dg.comparison_table(res_x)[["approach","val QLIKE","test QLIKE","test RMSE","val rank (QLIKE)","test rank (QLIKE)"]])""")

md("""## 11. Conclusions (see the report for full discussion)
1. Every forecasting model beat both baselines on the test period (QLIKE 31–47% lower than the 22-day historical mean).
2. Traditional and machine-learning models performed **similarly**; Ridge regression ranked first on both validation and test, but
   Diebold–Mariano tests do not show significant differences among the five models.
3. The ranking is **fragile**: it changes when the single 1 Dec 2025 shock is excluded (ARIMA first) and on the non-voting share (Random Forest first).
4. Tree models fit the training data far better than they forecast (train QLIKE ≈ 0.33–0.36 vs validation ≈ 0.66–0.68): overfitting with only ~250 training days.
5. No model anticipated the 1 Dec 2025 shock, which arrived in a *low*-volatility regime — the main practical limitation for risk management.""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/financial_time_series_analysis.ipynb")
print("notebook written")
