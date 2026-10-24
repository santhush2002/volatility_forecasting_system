# Forecasting next-day volatility of Commercial Bank of Ceylon (COMB.N0000)
### CM3023 Group Assignment – Real-World Financial Time Series Analytics

**Research question:** How accurately can the next trading day's volatility of COMB voting shares be forecast by
traditional time-series models (ARIMA on log-volatility, GARCH) compared with machine-learning models (Ridge,
Random Forest, Gradient Boosting), relative to simple baselines, one day ahead on an untouched test period
(Aug–Dec 2025)? Do the differences persist across calm and turbulent regimes?

## Folder structure
```
financial_time_series_project/
├── data/original/COMB_Stock_Data_2024_2025.xlsx   raw file, never modified
├── data/processed/COMB_N_clean.csv, COMB_X_clean.csv
├── notebooks/financial_time_series_analysis.ipynb  executed, step-by-step with explanations
├── src/
│   ├── config.py                paths, split dates, corporate actions
│   ├── data_processing.py       loading, quality checks, cleaning, adjusted returns, target
│   ├── feature_engineering.py   leakage-free features + chronological split
│   ├── exploratory_analysis.py  EDA figures, descriptive stats, ADF/KPSS/Ljung-Box/ARCH tests
│   ├── modelling.py             baselines, ARIMA, GARCH family, Ridge/RF/GBM
│   ├── evaluation.py            QLIKE, RMSE, MAE, MZ-R², Diebold-Mariano
│   ├── diagnostics.py           comparison tables, residual tests, forecast plots
│   ├── additional_analysis.py   regimes, rolling stability, shocks, importance
│   └── pipeline.py              runs the full model workflow for one share class
├── main.py                      regenerates every table/figure/prediction
├── check_leakage.py             automated look-ahead test
├── build_notebook.py            (re)creates the notebook
├── outputs/figures/             16 PNG figures (200 dpi)
├── outputs/tables/              30 CSV tables + all_tables.xlsx
├── outputs/predictions/         dated actual vs predicted values (all sets, and test only)
├── outputs/model_results/       summary.json, run log, leakage check
└── report/final_report.docx, final_report.pdf, findings_summary.md
```

## Installation (Windows, VS Code or Jupyter)
```bash
# 1. Python 3.10+ recommended. In a terminal opened in this folder:
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running
```bash
python main.py              # ~2 minutes; rewrites everything in outputs/
python check_leakage.py     # optional: proves forecasts don't use future data
jupyter notebook notebooks/financial_time_series_analysis.ipynb   # Kernel > Restart & Run All
```
To rebuild the report after changing results: `cd report && python build_report.py && node render_docx.js report_content.json final_report.docx`
(needs Node.js with `npm install docx`), then export to PDF from Word or LibreOffice.

## Key design decisions
* The Excel file contains **three securities** (N voting, X non-voting, R rights). N is analysed; X is a robustness check; R is excluded.
* The **OPEN column equals CLOSE on every 2025 day**; it is never used.
* Returns are **adjusted** for the FY2023 and FY2024 dividends and the July 2024 1-for-5 rights issue.
* **Target:** Parkinson range volatility σ = 100·|ln(H/L)|/√(4 ln 2), forecast one day ahead.
* **Split (target dates):** train 2024-02-07 → 2025-02-28 (254), validation 2025-03-03 → 2025-07-31 (99), test 2025-08-01 → 2025-12-31 (102).
* Model choice uses training (ARIMA AIC) and validation (QLIKE) only; final models re-fitted on train+validation and evaluated on test once.

## Headline results (test period, QLIKE – lower is better)
| Model | Test QLIKE | excl. 1 Dec 2025 |
|---|---|---|
| Ridge regression | 1.658 | 0.515 |
| GARCH(1,1)-t | 1.686 | 0.548 |
| Gradient boosting | 1.893 | 0.512 |
| ARIMA(1,0,1) on ln-vol | 1.937 | 0.504 |
| Random forest | 2.169 | 0.505 |
| 22-day historical mean (baseline) | 3.140 | 0.782 |
| Naive σ_t (baseline) | 4.594 | 0.973 |

All models beat both baselines; differences among the five models are not statistically significant (Diebold–Mariano).
See `report/final_report.pdf` for the full analysis.
