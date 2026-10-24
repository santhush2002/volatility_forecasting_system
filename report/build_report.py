"""
Builds report_content.json from the CSV outputs (single source of truth), then
render_docx.js turns it into final_report.docx. All numbers in tables and most
numbers in the text are read from outputs/tables, so the report cannot drift
from the code.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "outputs" / "tables"
SUM = json.load(open(HERE.parent / "outputs" / "model_results" / "summary.json"))


def T(name, **kw):
    return pd.read_csv(TAB / f"{name}.csv", **kw)


def f(x, d=3):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "–"
    return f"{x:,.{d}f}"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


B = []
add = B.append
P = lambda t: add({"t": "p", "text": t})
H1 = lambda t: add({"t": "h1", "text": t})
H2 = lambda t: add({"t": "h2", "text": t})
H3 = lambda t: add({"t": "h3", "text": t})
BOX = lambda t: add({"t": "box", "text": t})
BUL = lambda items: add({"t": "bullets", "items": items})
_num_i = [0]


def NUM(items):
    _num_i[0] += 1
    add({"t": "numbers", "items": items, "ref": ["num", "num2", "num3", "num4", "num5", "num6", "num7", "num8"][(_num_i[0] - 1) % 8]})


EQ = lambda t: add({"t": "eq", "text": t})
IMG = lambda p, cap, w=620: (add({"t": "img", "path": f"../outputs/figures/{p}", "width": w, "alt": cap}),
                             add({"t": "caption", "text": cap}))


def TABLE(caption, header, rows, widths=None, small=False, highlight=None):
    add({"t": "tcaption", "text": caption})
    add({"t": "table", "header": header, "rows": [[str(c) for c in r] for r in rows], "widths": widths,
         "small": small, "highlight": highlight or []})


# ---------------------------------------------------------------- load data
comp = T("T15_model_comparison", index_col=0)
imp_tab = T("T16_improvement", index_col=0)
dm_h = T("T17_dm_vs_hist", index_col=0)
dm_b = T("T18_dm_vs_bestval", index_col=0)
dm_hx = T("T22b_dm_vs_hist_excl_outlier", index_col=0)
sens = T("T22_outlier_sensitivity", index_col=0)
split = T("T04_split")
desc = T("T05_descriptive", index_col=0)
per = T("T06_period_stats", index_col=0)
stat = T("T07_stationarity", index_col=0)
dep = T("T08_dependence")
ca = T("T03_corporate_actions")
qual = T("T02_quality_N")
arsel = T("T10_arima_selection")
arpar = T("T11_arima_params", index_col=0)
gsel = T("T12_garch_selection")
gpar = T("T13_garch_params", index_col=0)
rtest = T("T19_residual_tests")
reg = T("T20_regimes")
regx = T("T20b_regimes_excl_outlier")
ev = T("T21_event_windows")
pimp = T("T23_permutation_importance")
rcoef = T("T24_ridge_coefficients")
robx = T("T25_robustness_X", index_col=0)
feats = T("T09_features")
tun = {k: T(f"T14_tuning_{k}") for k in ("ridge", "random", "gradient")}
raw_ov = T("T01_raw_overview")

AR, GA = SUM["arima_name"], SUM["garch_name"]
HM, NV = "Historical mean (22-day)", "Naive (sigma_t)"
models = list(comp.index)
trad = [m for m in models if comp.loc[m, "approach"] == "Traditional"]
ml = [m for m in models if comp.loc[m, "approach"] == "Machine learning"]
nonbase = trad + ml
best_val, best_test = SUM["best_val_model"], SUM["best_test_model"]
q = lambda m, s="test": comp.loc[m, f"{s} QLIKE"]
impr = imp_tab["QLIKE improvement vs 22-day mean (%)"]
best_excl = sens["test QLIKE (excl. 1 Dec)"].idxmin()
qa = dict(zip(arpar.index, arpar["estimate"]))
gp = dict(zip(gpar.index, gpar["estimate"]))
gpp = dict(zip(gpar.index, gpar["p-value"]))
dep_d = {(r.test, r.series): r for r in dep.itertuples()}
tr_n, va_n, te_n = split["observations"].tolist()
ml_gap = {m: (comp.loc[m, "train QLIKE"], comp.loc[m, "val QLIKE"]) for m in ml}

# ---------------------------------------------------------------- cover
add({"t": "title", "text": "General Sir John Kotelawala Defence University", "bold": True, "size": 30, "before": 1200})
add({"t": "title", "text": "BSc (Hons) Data Science and Business Analytics", "size": 24})
add({"t": "title", "text": "Module CM3023 – [module title as stated in the module handbook]", "size": 24})
add({"t": "title", "text": "Group Assignment: Real-World Financial Time Series Analytics", "size": 24, "before": 400})
add({"t": "title", "text": "Forecasting Next-Day Volatility of Commercial Bank of Ceylon PLC Shares:",
     "bold": True, "size": 34, "color": "1F4E79", "before": 600})
add({"t": "title", "text": "Traditional Time-Series Models versus Machine Learning on Colombo Stock Exchange Data, 2024–2025",
     "bold": True, "size": 28, "color": "1F4E79"})
add({"t": "title", "text": "Group members", "bold": True, "size": 24, "before": 800})
for i in range(1, 5):
    add({"t": "title", "text": f"[Name {i}]  –  [Registration number {i}]", "size": 22})
add({"t": "title", "text": "Submission date: [dd Month 2026]", "size": 22, "before": 500})
add({"t": "pb"})
add({"t": "TOCPLACE"})
add({"t": "pb"})

# ---------------------------------------------------------------- abstract
H1("1. Abstract")
P(f"Volatility – how much a share price can move in a day – drives margin requirements, position limits, Value-at-Risk and "
  f"the timing of large trades. This study asks whether the next trading day's volatility of Commercial Bank of Ceylon PLC "
  f"voting shares (COMB.N0000), the largest listed private bank on the Colombo Stock Exchange (CSE), can be forecast more "
  f"accurately with traditional time-series models or with machine-learning (ML) models. Using 478 daily CSE observations "
  f"(2 January 2024 – 31 December 2025), we measured daily volatility with the Parkinson high–low range estimator after "
  f"separating three securities mixed in the source file and correcting returns for two dividends and a 1-for-5 rights issue. "
  f"Two baselines (yesterday's volatility; 22-day historical mean), two traditional models ({AR}; {GA} with a scale "
  f"calibration) and three ML models (Ridge regression, Random Forest, Gradient Boosting on 14 lagged features) were compared "
  f"one day ahead with a strictly chronological split: training ({tr_n} days), validation ({va_n}) and an untouched test period "
  f"(August–December 2025, {te_n} days). On the test period every model beat both baselines, reducing QLIKE loss by "
  f"{impr[nonbase].min():.0f}–{impr[nonbase].max():.0f}% relative to the historical mean. Ridge regression ranked first on both "
  f"validation (QLIKE {q(best_val, 'val'):.3f}) and test (QLIKE {q(best_test):.3f}), but Diebold–Mariano tests found no "
  f"statistically significant difference between it and the other four models, and the ranking changed when one extreme day "
  f"(1 December 2025, after Cyclone Ditwah) was excluded ({best_excl} first) and when the identical pipeline was applied to "
  f"the non-voting share ({SUM['X_best_test']} first). The additional investigation of volatility regimes showed that the "
  f"models' advantage over the baselines held in both calm and turbulent periods, but that the largest losses came from a "
  f"shock arriving in a calm regime, which no model anticipated. We conclude that volatility of this stock is genuinely "
  f"forecastable from its own recent history and trading activity, that the information is largely linear and short-memory "
  f"(so flexible ML adds little over a well-specified ARIMA on log-volatility), and that the choice between model families "
  f"matters far less than choosing a sensible volatility measure and avoiding naive baselines.")

# ---------------------------------------------------------------- intro
H1("2. Introduction")
H2("2.1 Background and the real-world problem")
P("Commercial Bank of Ceylon PLC (ticker COMB) is one of the most heavily traded stocks on the CSE and a component of the "
  "S&P Sri Lanka 20 index. Between 2024 and 2025 its share price roughly doubled as Sri Lanka's economy recovered from the "
  "2022 crisis, but the path was not smooth: a Rs 22.5 billion rights issue was announced in May 2024, US reciprocal tariffs "
  "triggered a market-wide circuit breaker in April 2025, and Cyclone Ditwah struck the country at the end of November 2025.")
P("For a broker, fund manager, bank treasury or regulator, the most decision-relevant quantity is often not tomorrow's price "
  "direction – which is very hard to predict – but tomorrow's *risk*. A good one-day volatility forecast is the input to "
  "daily Value-at-Risk, to margin and collateral haircuts, to stop-loss distances and position sizing, and to the scheduling "
  "of large block trades in a market where liquidity is thin. Over-predicting risk ties up capital unnecessarily; "
  "under-predicting it exposes the institution to losses it was not prepared for.")
H2("2.2 Why volatility and not price or return?")
P(f"We initially considered forecasting prices or next-day returns. The data argued against both. Log prices are "
  f"non-stationary (ADF p = {stat.loc['log close price', 'ADF p']:.3f}; KPSS rejects stationarity), so a price forecast is "
  f"dominated by the random-walk component. Returns are close to unpredictable in magnitude and, even where weak "
  f"autocorrelation exists, a return forecast mostly measures noise. Daily volatility, in contrast, is strongly persistent: "
  f"the Ljung–Box Q(10) statistic for log range volatility is {dep_d[('Ljung-Box Q(10)', 'log Parkinson vol')].statistic:.1f} "
  f"(p < 0.001) in the training data. A well-posed forecasting problem with a clear financial use is therefore volatility.")
H2("2.3 Research question, aim and objectives")
BOX("**Research question.** How accurately can the next trading day's volatility of COMB voting shares be forecast by "
    "traditional time-series models (ARIMA on log-volatility and GARCH) compared with machine-learning models (Ridge "
    "regression, Random Forest, Gradient Boosting), relative to simple baselines, when evaluated one day ahead on an untouched "
    "final period (August–December 2025) – and do the differences persist across calm and turbulent market regimes?")
P("**Aim.** To produce a fair, leakage-free comparison of the two modelling traditions for daily equity risk forecasting on "
  "a frontier-market stock, and to explain – not only report – why they perform as they do.")
P("**Objectives.**")
NUM(["Build a clean, corporate-action-adjusted daily dataset for COMB and a defensible daily volatility measure.",
     "Characterise the statistical properties of returns and volatility (stationarity, autocorrelation, fat tails, clustering).",
     "Fit two baselines, two traditional and three ML models under one common forecasting protocol and select all "
     "specifications without using the test period.",
     "Compare out-of-sample accuracy with QLIKE, RMSE and MAE and test the significance of differences (Diebold–Mariano).",
     "Investigate whether performance differs between low- and high-volatility regimes and around documented market shocks."])
P("**Hypotheses.** H1: all fitted models beat the naive and historical-mean baselines. H2: ML models do *not* necessarily "
  "outperform traditional models; with about 250 training days we expected flexible models to overfit. H3: relative "
  "performance differs between calm and turbulent regimes.")
H2("2.4 Scope and contribution")
P("The study covers one stock, two years and a one-day horizon. Its contribution is methodological honesty on a small, "
  "real frontier-market dataset: it shows how data problems specific to CSE exports (mixed share classes, unadjusted prices, "
  "an unreliable open price) change the analysis, and it reports a result that is less dramatic but more useful than "
  "'machine learning wins'.")

# ---------------------------------------------------------------- data
H1("3. Dataset and Research Design")
H2("3.1 Source and contents")
P("The dataset is the file *COMB_Stock_Data_2024_2025.xlsx* supplied by the group. Its column layout (company ID, main "
  "type, sub type, short name, trading date, high, low, close, open, trade volume, share volume, turnover) matches the CSE "
  "company trade-summary format, so it is consistent with daily data downloaded from the Colombo Stock Exchange "
  "(www.cse.lk). We have not independently re-downloaded the series; the price path, the 7 April 2025 sell-off and the "
  "rights-issue reaction on 29 May 2024 agree with contemporaneous press reports (Daily Mirror, 2024; 2025).")
TABLE("Table 1. Securities found in the raw file (969 rows, 12 columns)",
      ["Main type", "Sub type", "Rows", "First date", "Last date", "Meaning"],
      [[r.share_class, r.sub_type, r.rows, r.start[:10], r.end[:10], r.meaning] for r in raw_ov.itertuples()],
      widths=[1, 1, 1, 1.5, 1.5, 4])
P("**What we did and why.** The file mixes three different securities. Voting (N) and non-voting (X) shares have "
  "different prices, liquidity and investor bases, and rights entitlements (R) are short-lived derivatives worth a few "
  "rupees. Pooling them would create a meaningless series. The voting share is the main analysis because it is far more "
  "liquid (median daily turnover about six times that of X); the non-voting share is used as a robustness check; rights are "
  "excluded.")
H2("3.2 Data-quality checks")
TABLE("Table 2. Data-quality summary, COMB.N0000 (before cleaning)", ["Check", "Result"],
      [[r.check, r.result] for r in qual.itertuples()], widths=[3, 2], small=True)
P("Three findings shaped the methodology. First, the **open price equals the close on every trading day of 2025** "
  "(55% of all days), whereas this happens on only about 11% of 2024 days; in 2025 the field is evidently not a genuine "
  "opening price in this export. We therefore never use the open price – which rules out Garman–Klass and similar "
  "estimators that need it. Second, four LOW values are missing (all in 2024). Third, there are no duplicates, no "
  "impossible prices (close outside the high–low range) and no zero-range days; gaps of 2–5 calendar days are weekends "
  "and CSE holidays, so the series is a complete trading-day calendar.")
H2("3.3 Preprocessing decisions")
BUL(["**Missing LOW (4 days):** replaced with min(open, close) of the same day. This uses no future information, but it "
     "can only understate the day's range; the days are flagged and all fall in the training period.",
     "**Outliers kept:** large moves such as the April 2025 tariff sell-off and 1 December 2025 are real market events; "
     "deleting them would make the models look better than they are. Their influence is analysed explicitly (Section 6.6).",
     "**Corporate actions:** unadjusted closing prices fall mechanically on ex-dividend (XD) and ex-rights (XR) days. "
     "Close-to-close returns on those days were corrected (Table 3). The XR date of 16 July 2024 is inferred from the data: "
     "the opening price that day (Rs 101.00) almost exactly equals the theoretical ex-rights price "
     "(5 × 104.50 + 85)/6 = Rs 101.25, and the rights entitlements traded from 30 July to 6 August 2024.",
     "**Original data kept:** the raw Excel file is stored unchanged in data/original; cleaned series are written to "
     "data/processed."])
EQ("Ex-dividend day:  r_t = ln[(C_t + D) / C_(t−1)]")
EQ("Ex-rights day:  r_t = ln(C_t / TERP),   TERP = (C_(t−1) + k·S) / (1 + k),   k = 1/5")
TABLE("Table 3. Corporate-action adjustments (voting share)",
      ["Date", "Event", "Raw return (%)", "Adjusted return (%)", "Note"],
      [[r.date, r.type, f(r._4, 2), f(r._5, 2), r.note] for r in ca.itertuples()], widths=[1.3, 1, 1.1, 1.2, 4.5], small=True)
P("The FY2024 dividend (Rs 7.50 cash + Rs 2.00 scrip, XD 1 April 2025) is confirmed by the CSE announcement. The FY2023 "
  "dividend amount (Rs 6.50 per share) is taken from a broker report and its cash/scrip split was not verified; because "
  "it affects only one return in the training sample, the impact on the results is negligible.")
H2("3.4 Target variable and forecast horizon")
P("The target is the **Parkinson (1980) range-based daily volatility**, in percent:")
EQ("σ_t = 100 × |ln(H_t / L_t)| / √(4 ln 2)")
P("We chose it over squared returns for three reasons. (i) It uses the whole intraday trading range, so it is roughly "
  "five times more statistically efficient than a squared close-to-close return as a measure of daily variance. "
  "(ii) It is unaffected by dividends and rights issues, which move the overnight price, not the intraday range. "
  "(iii) It needs only high and low prices, which are reliable in this dataset, unlike the open. The limitation is that it "
  "ignores overnight gaps and is affected by the CSE tick size (Rs 0.25 at these price levels), which makes very quiet "
  "days look discrete. **Horizon:** one trading day ahead. Each forecast for day t+1 is made after the close of day t, "
  "using only data up to day t – a realistic setting for a risk desk that sets limits for the next session.")
H2("3.5 Chronological evaluation design")
TABLE("Table 4. Chronological split (dates are forecast-target dates)",
      ["Set", "First date", "Last date", "Days", "Share", "Used for"],
      [list(r)[1:] for r in split.itertuples()], widths=[1.2, 1.3, 1.3, 0.8, 0.8, 3])
P("The first 23 trading days are consumed by 22-day rolling features, so the modelling sample has 455 forecast days. "
  "A 70/15/15 split would leave a test period of only about 68 days; we used calendar boundaries giving roughly "
  "56/22/22% so that the validation period contains the April 2025 shock and the test period spans five full months. "
  "The protocol, applied identically to every model, is:")
NUM(["Estimate parameters on the training set only.",
     "Forecast every validation day one step ahead with those fixed parameters; choose orders and hyper-parameters by "
     "validation QLIKE.",
     "Re-estimate each chosen specification once on training + validation data.",
     "Forecast each test day one step ahead with fixed parameters. New observations are fed in as they occur (as a risk "
     "desk would), but nothing is re-tuned and the test period is never used for any decision."])
P("Training performance is in-sample fit; validation performance guided choices and is therefore slightly optimistic; "
  "test performance is the only unbiased estimate of real forecasting ability and is the basis of our conclusions.")

# ---------------------------------------------------------------- EDA
H1("4. Exploratory Data Analysis")
H2("4.1 Descriptive statistics")
TABLE("Table 5. Descriptive statistics, COMB.N0000 (full sample)",
      ["Variable", "n", "Mean", "Std", "Min", "Median", "Max", "Skew", "Ex. kurt."],
      [[i, int(r.n), f(r["mean"], 3 if "Volume" not in i else 0), f(r["std"], 3 if "Volume" not in i else 0), f(r["min"], 3 if "Volume" not in i else 0),
        f(r["median"], 3 if "Volume" not in i else 0), f(r["max"], 3 if "Volume" not in i else 0), f(r["skewness"], 2), f(r["excess kurtosis"], 2)]
       for i, r in desc.iterrows()], widths=[3, 0.7, 1.2, 1.2, 1, 1.2, 1.3, 0.8, 0.9], small=True)
TABLE("Table 6. Return and volatility by period (price change is unadjusted)",
      ["Period", "Days", "Price change (%)", "Mean ret. (%/day)", "Std ret. (%/day)", "Mean vol (%)", "Median vol (%)", "Annualised vol (%)"],
      [[i, int(r.days), f(r.iloc[1], 1), f(r.iloc[2], 3), f(r.iloc[3], 3), f(r.iloc[4], 3), f(r.iloc[5], 3), f(r.iloc[6], 1)] for i, r in per.iterrows()],
      widths=[2, 0.7, 1.1, 1.1, 1.1, 1, 1, 1.1], small=True)
P(f"Average daily range volatility is {desc.iloc[2]['mean']:.2f}% with a median of {desc.iloc[2]['median']:.2f}%, and "
  f"the distribution is highly right-skewed (skewness {desc.iloc[2]['skewness']:.1f}). Importantly for the comparison, "
  f"**the test period was calmer than the training period** (median volatility {per.loc['Test period', 'median Parkinson vol (%)']:.2f}% "
  f"versus {per.loc['Training period', 'median Parkinson vol (%)']:.2f}%), apart from one extreme day. Models estimated on "
  f"a more volatile history will tend to over-predict in a calm period – something the QLIKE loss penalises less than "
  f"under-prediction.")
H2("4.2 Price, returns and volatility clustering")
IMG("fig01_price_history.png", "Figure 1. Closing prices of the voting and non-voting shares with ex-dividend (XD) and "
    "ex-rights (XR) dates and the training (blue), validation (orange) and test (green) periods. Interpretation: a strong "
    "upward trend means price levels are non-stationary; the corporate-action dates are where unadjusted prices drop mechanically.")
IMG("fig02_returns_rolling.png", "Figure 2. Adjusted daily returns (top) and rolling 22-day mean and standard deviation "
    "(bottom). Interpretation: the rolling standard deviation varies between roughly 0.6% and 3.6%, and large moves "
    "come in bursts (May 2024, April 2025) – the visual signature of time-varying volatility.")
IMG("fig03_return_distribution.png", "Figure 3. Distribution and normal Q-Q plot of daily returns. Interpretation: far "
    f"more mass in the tails than a normal distribution (excess kurtosis {desc.iloc[1]['excess kurtosis']:.1f}); this "
    "motivates the Student-t distribution in the GARCH models.", 600)
H2("4.3 The forecasting target")
IMG("fig04_target_series.png", "Figure 4. Daily Parkinson volatility with its 22-day mean (top) and on the log scale "
    "(bottom). Interpretation: volatility moves in persistent waves around a stable level; ln-volatility looks stationary. "
    "The spike on 1 December 2025 (intraday low Rs 170 against a close of Rs 200.25) is the largest value in the sample.")
IMG("fig05_target_distribution.png", "Figure 5. Distributions of volatility and ln-volatility, and volatility by weekday "
    f"(training). Interpretation: the log transform makes the target close to symmetric; there is no weekday pattern "
    f"(Kruskal–Wallis H = {SUM['weekday_kruskal']['H']:.2f}, p = {SUM['weekday_kruskal']['p']:.2f}), so a seasonal model is not justified.")
P("The 1 December 2025 observation deserves comment. It was the first trading session after Cyclone Ditwah, and the CSE "
  "was under pressure in late 2025 (The Morning, 2025). However, the voting share's low of Rs 170 (−18% intraday, closing only "
  "−3.7%) is far deeper than the non-voting share's low that day (−4.2%). It may reflect one or a few trades at extreme "
  "prices. We cannot verify this from the data, so we keep the value and report all key results with and without it.")
H2("4.4 Statistical tests")
TABLE("Table 7. Stationarity tests on the training sample (ADF H0: unit root; KPSS H0: stationary)",
      ["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p", "Conclusion (5%)"],
      [[i, f(r.iloc[0], 2), pv(r.iloc[1]), f(r.iloc[2], 3), ("≥" if r.iloc[3] >= 0.1 else "≤" if r.iloc[3] <= 0.01 else "") + f(r.iloc[3], 2), r.iloc[4]] for i, r in stat.iterrows()],
      widths=[2.2, 1, 0.9, 1, 0.9, 2.6])
TABLE("Table 8. Dependence and distribution tests on the training sample",
      ["Test", "Series", "Statistic", "p-value", "Reject H0 at 5%?"],
      [[r.test, r.series, f(r.statistic, 2), pv(r._4), r._5] for r in dep.itertuples()], widths=[1.8, 1.8, 1, 1, 1.2])
P("KPSS p-values are reported by statsmodels only within 0.01–0.10, hence the inequality signs. The tests lead to four "
  "modelling decisions:")
BUL(["**Do not model prices.** The log price has a unit root.",
     "**Model ln-volatility without differencing (d = 0).** ADF and KPSS agree that it is stationary.",
     "**Volatility is forecastable.** Ln-volatility has very strong autocorrelation (Ljung–Box Q(10) p < 0.001; Figure 6c–d: "
     "the ACF decays slowly and the PACF has two to three significant lags, suggestive of an ARMA(1,1)-type process).",
     f"**A warning for GARCH.** Squared returns show no significant autocorrelation (Q(10) p = "
     f"{pv(dep_d[('Ljung-Box Q(10)', 'squared return')]._4)}) and the ARCH-LM test is not significant (p = "
     f"{pv(dep_d[('ARCH-LM (5 lags)', 'adjusted return')]._4)}). Volatility clustering is present in the range measure but "
     "invisible in squared daily returns, which are a very noisy volatility proxy. Because GARCH learns only from squared "
     "returns, we expected it to estimate the dynamics imprecisely."])
P(f"Returns themselves are autocorrelated (Ljung–Box Q(5) p = {pv(dep_d[('Ljung-Box Q(5)', 'adjusted return')]._4)}), a "
  "common feature of thinly traded frontier markets (stale prices and gradual price adjustment). This is noted but is not "
  "the object of forecasting here.")
IMG("fig06_acf_pacf.png", "Figure 6. ACF of returns and squared returns, and ACF/PACF of ln-volatility (training sample, "
    "95% bands). Interpretation: strong persistence in ln-volatility (c, d) contrasts with almost none in squared returns (b).")

# ---------------------------------------------------------------- methodology
H1("5. Methodology")
H2("5.1 Common forecasting protocol")
P("All seven methods forecast the same quantity (σ for day t+1, in percent), from the same origin (close of day t), "
  "for the same validation and test dates. Models fitted on ln σ are converted back with Duan's (1983) smearing factor "
  "estimated on the fitting sample, because exp(E[ln σ]) systematically under-estimates E[σ].")
H2("5.2 Baselines (defined before modelling)")
BUL(["**Naive (random walk):** σ̂_(t+1) = σ_t. The simplest possible forecast.",
     "**22-day historical mean:** σ̂_(t+1) = mean(σ_(t−21), …, σ_t). The 'historical volatility' that practitioners commonly "
     "use; it is our primary benchmark because it is what a desk without a model would actually use."])
H2("5.3 Traditional model 1 – ARIMA on ln-volatility")
P(f"Because ln σ is stationary and its ACF/PACF suggest a low-order ARMA, we fitted ARIMA(p, 0, q) with a constant for all "
  f"p, q ≤ 3 on the training data and selected by AIC. The chosen model is **{AR}** (AIC {arsel.loc[0, 'AIC']:.1f}; "
  f"the next best, ARIMA(1,0,2), is {arsel.loc[1, 'AIC'] - arsel.loc[0, 'AIC']:.1f} points worse and BIC also favours (1,0,1)). "
  "ARIMA assumes linear dynamics and constant error variance in ln σ; it cannot use volume or calendar information. "
  "All fits converged without warnings. Prediction intervals are formed on the log scale and exponentiated; because exp is "
  "monotonic, the quantiles transform exactly.")
TABLE("Table 9. ARIMA(1,0,1) parameters, re-estimated on training + validation data",
      ["Parameter", "Estimate", "Std. error", "p-value"],
      [[i, f(r.estimate, 4), f(r.iloc[1], 4), pv(r.iloc[2])] for i, r in arpar.iterrows()], widths=[2, 1, 1, 1])
P(f"The AR coefficient ({qa['ar.L1']:.2f}) shows that volatility shocks persist; the negative MA term "
  f"({qa['ma.L1']:.2f}) means part of each day's surprise is noise that does not carry over – exactly what one expects "
  "when the 'true' volatility is observed with error through a daily range.")
H2("5.4 Traditional model 2 – GARCH family")
P("GARCH models are the standard econometric tool for volatility clustering (Engle, 1982; Bollerslev, 1986). We estimated "
  "three variants with a constant mean and Student-t errors on the corporate-action-adjusted returns: GARCH(1,1), "
  "GJR-GARCH(1,1) (asymmetric response to negative returns; Glosten et al., 1993) and EGARCH(1,1) (Nelson, 1991). "
  "The variant with the lowest validation QLIKE was kept.")
TABLE("Table 10. GARCH variant selection (parameters estimated on training data)",
      ["Specification", "Train log-lik.", "Train AIC", "Train BIC", "Validation QLIKE"],
      [[r.specification, f(r._2, 1), f(r._3, 1), f(r._4, 1), f(r._5, 4)] for r in gsel.itertuples()], widths=[2, 1, 1, 1, 1.2])
P(f"GARCH conditional volatility measures close-to-close variability, which is not on the same scale as the intraday range. "
  f"To make the comparison fair we mapped it onto the target with a regression on logs, ln σ_t = a + b·ln h_t^½, estimated "
  f"**on the fitting sample only** (test fit: a = {SUM['garch_calibration_test']['calib_a']:.3f}, b = "
  f"{SUM['garch_calibration_test']['calib_b']:.3f}). The near-unit slope shows the two measures move proportionally.")
TABLE(f"Table 11. {GA} parameters, re-estimated on training + validation returns",
      ["Parameter", "Estimate", "Std. error", "p-value"],
      [[i, f(r.estimate, 4), f(r.iloc[1], 4), pv(r.iloc[2]) if not pd.isna(r.iloc[2]) else "–"] for i, r in gpar.iterrows()], widths=[2.4, 1, 1, 1])
P(f"The ARCH coefficient α = {gp['alpha[1]']:.2f} is significant (p = {gpp['alpha[1]']:.3f}) but β = {gp['beta[1]']:.2f} "
  f"is only marginally significant (p = {gpp['beta[1]']:.3f}) and persistence is a modest {gp['persistence (alpha+beta+gamma/2)']:.2f}, "
  f"far below the 0.95–0.99 typical of liquid markets. The tail parameter ν = {gp['nu']:.1f} confirms very fat tails. "
  "This is consistent with the weak ARCH evidence in Section 4.4.")
H2("5.5 Feature engineering for machine learning")
P("Fourteen predictors were built, each from information available at the close of day t (implemented by computing the "
  "quantity on day t and shifting it one row forward). Rolling windows look backward only. The only information about "
  "day t+1 itself is calendar information published in advance by the CSE (number of days until the next session and whether "
  "it is a Monday).")
TABLE("Table 12. Features used by the ML models", ["Feature", "Meaning and expected role"],
      [[r.feature, r.description] for r in feats.itertuples()], widths=[1.3, 5], small=True)
P("The first five features form a 'HAR' structure (Corsi, 2009): daily, weekly and monthly volatility components. Absolute "
  "and signed returns capture the size and direction of the last move (negative returns often raise future volatility). "
  "Volume and number of trades proxy information arrival. All learned preprocessing (the Ridge scaler) is fitted inside a "
  "scikit-learn Pipeline on training rows only.")
IMG("fig07_feature_correlation.png", "Figure 7. Correlations among features and the next-day ln-volatility target (training "
    "sample). Interpretation: recent daily and weekly volatility, recent absolute returns and the number of trades are most correlated with "
    "the target (about 0.35–0.39); the lagged volatility features are highly correlated with each other, which motivates "
    "regularisation and explains why individual coefficients must be interpreted cautiously.", 520)
H2("5.6 Machine-learning models and hyper-parameter selection")
BUL(["**Ridge regression** (Hoerl and Kennard, 1970): a linear model with an L2 penalty, appropriate because the features are "
     "correlated and the sample is small. Grid: α ∈ {0.01, 0.1, 1, 10, 30, 100, 300}; features standardised.",
     "**Random Forest** (Breiman, 2001): averages 500 decision trees, can capture non-linear effects and interactions "
     "(e.g. volume mattering more after a large move). Grid: max depth {3, 6, none} × min leaf size {5, 15} × "
     "share of features per split {0.33, 1.0}.",
     "**Gradient Boosting** (Friedman, 2001): sequentially corrects errors with shallow trees; strong on tabular data but "
     "prone to overfitting. Grid: learning rate {0.02, 0.05} × trees {100, 300} × depth {2, 3}; subsample 0.8."])
P("Each grid point was fitted on the training data and scored by validation QLIKE – a deliberately small search "
  "(7, 12 and 8 settings) so that the validation period is not over-fitted. Neural networks/LSTM were not used: with about "
  "250 training observations they would have far more parameters than data points.")
TABLE("Table 13. Selected hyper-parameters", ["Model", "Selected setting", "Validation QLIKE"],
      [["Ridge regression", "α = 0.01 (smallest in grid)", f(tun['ridge'].loc[0, 'val_QLIKE'], 4)],
       ["Random forest", "max depth 6, min leaf 5, 33% of features per split", f(tun['random'].loc[0, 'val_QLIKE'], 4)],
       ["Gradient boosting", "learning rate 0.05, 100 trees, depth 2", f(tun['gradient'].loc[0, 'val_QLIKE'], 4)]],
      widths=[1.5, 3.5, 1.2])
P(f"Ridge chose the weakest penalty, and validation QLIKE barely changes between α = 0.01 and α = 1 "
  f"({tun['ridge'].loc[0, 'val_QLIKE']:.4f} vs {tun['ridge'].loc[2, 'val_QLIKE']:.4f}): the model is effectively an ordinary "
  "least-squares 'HAR-X' regression. Extending the grid further would not change the forecasts materially.")
H2("5.7 Evaluation metrics")
BUL(["**QLIKE (primary):** L = σ²/σ̂² − ln(σ²/σ̂²) − 1, computed on variances. Zero for a perfect forecast; lower is better; "
     "unit-free. Patton (2011) showed that QLIKE ranks forecasts correctly even when true volatility is only observed through a noisy "
     "proxy; it also penalises under-prediction of risk more than over-prediction, matching a risk manager's priorities. "
     "Limitation: very quiet days (tiny ranges) can produce large values for every model.",
     "**RMSE:** √mean(σ − σ̂)², in percentage points of daily volatility. Penalises large errors heavily, so it is sensitive to single extreme days.",
     "**MAE:** mean |σ − σ̂|, in percentage points; robust to outliers and easy to interpret.",
     "**Mincer–Zarnowitz R²:** R² from regressing actual on forecast volatility; shows how much variation the forecast tracks, ignoring bias.",
     "**Diebold–Mariano test** (Diebold and Mariano, 1995, with the Harvey et al., 1997 small-sample correction): tests H0 that two "
     "models have equal expected loss. A negative statistic means the first model has lower loss.",
     "**Relative improvement** = 100 × (baseline loss − model loss) / baseline loss. MAPE was not used because it explodes "
     "on days with very small volatility."])
H2("5.8 Leakage prevention (summary)")
BUL(["Chronological split; no random shuffling anywhere.",
     "Every feature is lagged by one trading day; rolling windows end at day t.",
     "ADF/KPSS, ARIMA order selection, the regime threshold and all scalers use training data only.",
     "Hyper-parameters and the GARCH variant are chosen on validation data; the test period is used exactly once.",
     "Calibration and smearing factors are estimated on the fitting sample, never on the forecast period.",
     "Corporate-action corrections use announced dividend/rights terms, which were public before the event."])
H2("5.9 Additional investigation")
P(f"The primary additional investigation is **performance across volatility regimes and over time**. Each out-of-sample "
  f"day (validation + test, 201 days) is classified at the forecast origin as *high-volatility* if the trailing 22-day "
  f"mean volatility exceeds its training-sample median ({SUM['regime_threshold_vol_pct']:.2f}% per day), otherwise "
  "*low-volatility*. The threshold is fixed from training data, so regime labels never use future information. We "
  "complement this with rolling 22-day losses, two externally documented shock windows, a sensitivity analysis for the "
  "1 December 2025 observation, permutation feature importance and a full re-run on the non-voting share.")

# ---------------------------------------------------------------- results
H1("6. Results")
H2("6.1 Validation and test performance")
rows = []
for m in models:
    r = comp.loc[m]
    rows.append([m, r.approach, f(r["train QLIKE"]), f(r["val QLIKE"]), f(r["test QLIKE"]), f(r["val RMSE"]), f(r["test RMSE"]),
                 f(r["test MAE"]), f(r["test MZ R2"]), int(r["val rank (QLIKE)"]), int(r["test rank (QLIKE)"])])
TABLE("Table 14. Model comparison – one-day-ahead forecasts (QLIKE primary; RMSE/MAE in percentage points)",
      ["Model", "Approach", "Train QLIKE", "Val QLIKE", "Test QLIKE", "Val RMSE", "Test RMSE", "Test MAE", "Test MZ R²", "Val rank", "Test rank"],
      rows, widths=[2.2, 1.4, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6], small=True,
      highlight=[models.index(best_test)])
P(f"**Validation.** Ridge regression had the lowest validation QLIKE ({q('Ridge regression', 'val'):.3f}), followed by "
  f"{AR} ({q(AR, 'val'):.3f}); both baselines were worst (naive {q(NV, 'val'):.3f}; historical mean {q(HM, 'val'):.3f}).")
P(f"**Test.** Ridge regression again ranked first ({q('Ridge regression'):.3f}), followed by {GA} ({q(GA):.3f}). The "
  f"baselines were last (historical mean {q(HM):.3f}, naive {q(NV):.3f}). All QLIKE values are much higher on the test "
  f"set than on validation – not because the models deteriorated in general but because of a single day: on 1 December "
  f"2025 the actual volatility was 11.2%, roughly ten times a typical forecast. RMSE is almost identical for the five fitted "
  f"models ({comp.loc[nonbase, 'test RMSE'].min():.3f}–{comp.loc[nonbase, 'test RMSE'].max():.3f}) because that day alone "
  f"contributes about 1.0 to the test mean squared error of every model.")
TABLE("Table 15. Relative improvement over the baselines on the test period",
      ["Model", "QLIKE vs 22-day mean (%)", "RMSE vs 22-day mean (%)", "QLIKE vs naive (%)"],
      [[i, f(r.iloc[0], 1), f(r.iloc[1], 1), f(r.iloc[2], 1)] for i, r in imp_tab.iterrows()], widths=[2.5, 1.4, 1.4, 1.4])
TABLE("Table 15b. Model summary: diagnostics and short interpretation",
      ["Model", "Key diagnostics", "Interpretation"],
      [[NV, "No parameters", "Reacts fully to one day's noise; worst on test (largest losses on calm days)."],
       [HM, "No parameters", "Smooth but slow; over-predicts after bursts and under-predicts at their start."],
       [AR, "Residual Ljung–Box p > 0.8; AR 0.77, MA −0.44; tiny train–val gap", "Captures the linear persistence of ln-volatility; best on test when 1 Dec is excluded."],
       [GA, "β only marginally significant; persistence 0.73; ν ≈ 3.8; std. residuals autocorrelated", "Learns from noisy squared returns; weakest fitted model on validation; test rank helped by higher average forecasts."],
       ["Ridge regression", "α at grid minimum (≈ OLS); train QLIKE ≈ validation QLIKE", "Linear HAR-type model plus activity variables; best on validation and on all-day test; not significantly better than the others."],
       ["Random forest", "Train QLIKE about half of validation QLIKE", "Overfits ~250 training days; competitive once the outlier is removed and on the X share."],
       ["Gradient boosting", "Train QLIKE about half of validation QLIKE", "Overfits but reacts well to shocks (best in the April 2025 tariff window)."]],
      widths=[1.8, 2.6, 3.2], small=True)
H2("6.2 Are the differences statistically significant?")
TABLE("Table 16. Diebold–Mariano tests on the test period (negative statistic = lower loss than the reference)",
      ["Model", "vs 22-day mean: DM (QLIKE)", "p", "vs 22-day mean: DM (sq. err.)", "p",
       f"vs {best_val}: DM (QLIKE)", "p"],
      [[m, f(dm_h.loc[m, "DM stat (QLIKE)"], 2), pv(dm_h.loc[m, "p (QLIKE)"]), f(dm_h.loc[m, "DM stat (squared error)"], 2),
        pv(dm_h.loc[m, "p (squared error)"]),
        f(dm_b.loc[m, "DM stat (QLIKE)"], 2) if m in dm_b.index else "–", pv(dm_b.loc[m, "p (QLIKE)"]) if m in dm_b.index else "–"]
       for m in models if m != HM], widths=[2.2, 1.1, 0.7, 1.1, 0.7, 1.1, 0.7], small=True)
P(f"With all {te_n} test days, no model beats the historical mean significantly under QLIKE (all p > 0.17), because the "
  f"loss differences are dominated by the 1 December outlier, which inflates their variance. Under squared error, ARIMA "
  f"(p = {dm_h.loc[AR, 'p (squared error)']:.3f}) and Random Forest (p = {dm_h.loc['Random forest', 'p (squared error)']:.3f}) "
  f"are significantly better than the historical mean at 5%. Against Ridge – the best model on validation – no other model, "
  "not even a baseline, differs significantly (all p > 0.2). When the outlier is excluded (Section 6.6), ARIMA, Random Forest and Gradient Boosting "
  "beat the historical mean significantly under QLIKE.")
H2("6.3 Forecast plots")
IMG(f"fig08_full_forecast_N.png", "Figure 8. Full sample with split periods, out-of-sample Ridge forecasts, the 22-day "
    "mean baseline and the ARIMA 80% prediction interval for the test period. Interpretation: the model forecasts react "
    "within a day to volatility bursts (April 2025), whereas the historical mean responds slowly and stays high after the burst has faded.")
IMG("fig15a_test_forecasts_traditional_N.png", "Figure 9. Test-period forecasts of the traditional models against actual "
    "volatility and the baseline. Interpretation: ARIMA and GARCH track the level and the persistence of spikes; GARCH reacts more "
    "strongly to large returns and sits higher on average.")
IMG("fig15b_test_forecasts_machine_N.png", "Figure 10. Test-period forecasts of the ML models. Interpretation: the three ML "
    "forecasts are very similar to each other and to ARIMA; none anticipated the 1 December spike, and all over-predicted "
    "the quiet days of late December.")
P(f"Mean test forecasts ({comp.loc[nonbase, 'test mean forecast'].min():.2f}–{comp.loc[nonbase, 'test mean forecast'].max():.2f}%) "
  f"exceed the mean actual volatility on all test days except the outlier (0.88%), i.e. the models systematically "
  "over-predicted risk in the calm test period – the consequence of estimating on a more volatile history. In risk-management "
  "terms this is the conservative direction of error.")
H2("6.4 Model diagnostics")
TABLE("Table 17. Ljung–Box tests on residuals of the traditional models (training sample)",
      ["Model", "Residual", "Lag", "LB stat", "p-value", "Autocorrelation left?"],
      [[r.model, r.residual, r.lag, f(r._4, 2), pv(r._5), r._6] for r in rtest.itertuples()], widths=[2, 2, 0.5, 0.9, 0.9, 1.2], small=True)
IMG("fig09_arima_residuals_N.png", "Figure 11. ARIMA(1,0,1) residual diagnostics. Interpretation: residuals show no "
    "remaining autocorrelation (Ljung–Box p > 0.8) and are roughly symmetric – the model has captured the linear dynamics of ln-volatility.")
IMG("fig10_garch_diagnostics_N.png", "Figure 12. GARCH(1,1)-t diagnostics. Interpretation: squared standardised residuals "
    "have no remaining ARCH structure and the Student-t fits the tails, but standardised residuals are autocorrelated "
    "(p < 0.01) because the constant-mean equation ignores the return autocorrelation found in Section 4.4.")
IMG("fig11_ml_diagnostics_N.png", "Figure 13. ML test diagnostics: actual vs forecast (top) and errors over time (bottom). "
    "Interpretation: forecasts are compressed into a narrow band (roughly 0.7–2.8%) while actual values range much more widely; "
    "errors are negative (over-prediction) on most quiet days and large and positive on shock days.")
P(f"A clear generalisation gap appears for the tree models: Random Forest's training QLIKE ({ml_gap['Random forest'][0]:.3f}) "
  f"is about half its validation QLIKE ({ml_gap['Random forest'][1]:.3f}), and similarly for Gradient Boosting "
  f"({ml_gap['Gradient boosting'][0]:.3f} vs {ml_gap['Gradient boosting'][1]:.3f}). Ridge and ARIMA show almost no gap "
  f"({comp.loc['Ridge regression', 'train QLIKE']:.3f} vs {q('Ridge regression', 'val'):.3f}; "
  f"{comp.loc[AR, 'train QLIKE']:.3f} vs {q(AR, 'val'):.3f}). The trees memorised training noise that did not recur.")
H2("6.5 Additional investigation: volatility regimes, stability and shocks")
piv = reg.pivot(index="model", columns="regime", values="QLIKE").loc[models]
pivx = regx.pivot(index="model", columns="regime", values="QLIKE").loc[models]
rk = regx.pivot(index="model", columns="regime", values="rank in regime (QLIKE)").loc[models]
nd = reg.groupby("regime")["days"].first()
TABLE("Table 18. Out-of-sample QLIKE by volatility regime (validation + test)",
      ["Model", f"Low-vol, all ({nd['Low-volatility']} d)", f"High-vol ({nd['High-volatility']} d)", "Low-vol excl. 1 Dec", "Rank low (excl.)", "Rank high"],
      [[m, f(piv.loc[m, "Low-volatility"]), f(piv.loc[m, "High-volatility"]), f(pivx.loc[m, "Low-volatility"]),
        int(rk.loc[m, "Low-volatility"]), int(rk.loc[m, "High-volatility"])] for m in models], widths=[2.4, 1.2, 1.2, 1.2, 0.9, 0.9], small=True)
IMG("fig12_regime_qlike_N.png", "Figure 14. QLIKE by regime with and without 1 December 2025. Interpretation: excluding "
    "the single shock, all five models have similar losses in both regimes and clearly lower losses than both the "
    "baselines; the naive forecast is worst in calm periods and the historical mean worst in turbulent ones.")
P(f"Four findings emerge. (1) **The models' advantage is not regime-specific.** In both regimes all five fitted models beat "
  f"both baselines. (2) **Baselines fail in opposite regimes.** The naive forecast is worst in calm periods (QLIKE "
  f"{pivx.loc[NV, 'Low-volatility']:.2f}), where single-day noise dominates, and the historical mean is worst in turbulent "
  f"periods ({piv.loc[HM, 'High-volatility']:.2f}), where it reacts too slowly. (3) **GARCH is relatively weakest in the "
  f"high-volatility regime** among fitted models ({piv.loc[GA, 'High-volatility']:.3f} vs {piv.loc['Gradient boosting', 'High-volatility']:.3f} "
  f"for Gradient Boosting), consistent with its imprecise dynamics. (4) **The worst losses come from a shock arriving in a "
  f"calm regime:** the 1 December spike fell in the low-volatility regime, and including it multiplies every model's "
  f"low-regime loss roughly three-fold.")
IMG("fig13_rolling_qlike_N.png", "Figure 15. Rolling 22-day QLIKE over the out-of-sample period (log scale). "
    "Interpretation: the five fitted models move together and stay below the baselines most of the time; the ordering "
    "among them changes from month to month, so no single model is consistently best.")
evp = ev.pivot(index="model", columns="event", values="QLIKE").loc[models]
evm = ev.pivot(index="model", columns="event", values="mean forecast").loc[models]
eva = ev.groupby("event")["mean actual"].first()
TABLE("Table 19. Performance in two documented shock windows",
      ["Model", "Tariff shock 2–11 Apr 2025: QLIKE", "mean forecast (%)", "Post-cyclone 1–5 Dec 2025: QLIKE", "mean forecast (%)"],
      [[m, f(evp.loc[m, "US tariff shock (2-11 Apr 2025)"], 2), f(evm.loc[m, "US tariff shock (2-11 Apr 2025)"], 2),
        f(evp.loc[m, "Post-Cyclone Ditwah (1-5 Dec 2025)"], 2), f(evm.loc[m, "Post-Cyclone Ditwah (1-5 Dec 2025)"], 2)] for m in models]
      + [["Actual mean volatility", "", f(eva['US tariff shock (2-11 Apr 2025)'], 2), "", f(eva['Post-Cyclone Ditwah (1-5 Dec 2025)'], 2)]],
      widths=[2.3, 1.3, 1.1, 1.3, 1.1], small=True)
P(f"During the April 2025 tariff sell-off, when the S&P SL20 triggered a circuit breaker (Daily Mirror, 2025), Gradient "
  f"Boosting had the lowest loss, followed by the naive forecast and Ridge – the models that react fastest to the previous day's large move; "
  f"the 22-day mean, still anchored to calmer March data, had a loss three to four times larger. After Cyclone Ditwah, every "
  f"model under-predicted the first day badly (forecasts near 1% against 11.2%) and most then over-predicted the following days; GARCH and Ridge had the lowest losses in that window.")
H2("6.6 Sensitivity to the 1 December 2025 observation")
TABLE("Table 20. Test results with and without 1 December 2025",
      ["Model", "QLIKE (all)", "QLIKE (excl.)", "RMSE (all)", "RMSE (excl.)", "Rank (all)", "Rank (excl.)"],
      [[i, f(r.iloc[0]), f(r.iloc[1]), f(r.iloc[2]), f(r.iloc[3]), int(r.iloc[4]), int(r.iloc[5])] for i, r in sens.iterrows()], widths=[2.4, 1, 1, 1, 1, 0.8, 0.8], small=True)
P(f"Excluding one day out of {te_n} changes the ranking: {best_excl} moves from fourth to first, Random Forest from fifth to "
  f"second, and GARCH from second to fifth; Ridge drops to fourth. The spread between the top four models is tiny "
  f"({sens['test QLIKE (excl. 1 Dec)'][nonbase].sort_values().iloc[0]:.3f}–{sens['test QLIKE (excl. 1 Dec)'][nonbase].sort_values().iloc[3]:.3f}), "
  f"while every model is {100*(1-sens['test QLIKE (excl. 1 Dec)'][nonbase].max()/sens.loc[HM,'test QLIKE (excl. 1 Dec)']):.0f}–{100*(1-sens['test QLIKE (excl. 1 Dec)'][nonbase].min()/sens.loc[HM,'test QLIKE (excl. 1 Dec)']):.0f}% better than the historical mean. In Diebold–Mariano tests without the outlier, ARIMA "
  f"(p = {dm_hx.loc[AR, 'p (QLIKE)']:.3f}), Random Forest (p = {dm_hx.loc['Random forest', 'p (QLIKE)']:.3f}) and Gradient "
  f"Boosting (p = {dm_hx.loc['Gradient boosting', 'p (QLIKE)']:.3f}) beat the historical mean at 5%; Ridge is borderline "
  f"(p = {dm_hx.loc['Ridge regression', 'p (QLIKE)']:.3f}) and GARCH is not (p = {dm_hx.loc[GA, 'p (QLIKE)']:.3f}). "
  "Excluding the day is *not* our main result – it would be cherry-picking – but it shows how fragile any 'winner' is.")
H2("6.7 What drives the ML forecasts? Feature importance")
pp = pimp.pivot(index="feature", columns="model", values="mean QLIKE increase")
pp = pp.loc[pp.mean(axis=1).sort_values(ascending=False).index]
TABLE("Table 21. Permutation importance: increase in test QLIKE when a feature is shuffled (test excl. 1 Dec; 30 repeats)",
      ["Feature", "Ridge", "Random forest", "Gradient boosting"],
      [[i, f(r["Ridge regression"], 4), f(r["Random forest"], 4), f(r["Gradient boosting"], 4)] for i, r in pp.iterrows()],
      widths=[2, 1, 1, 1], small=True)
IMG("fig14_permutation_importance_N.png", "Figure 16. Permutation importance by model. Interpretation: all three models "
    "rely mainly on the last two days' volatility, the number of trades and the latest (absolute and signed) return; "
    "weekly/monthly averages, momentum and calendar variables add almost nothing out of sample.")
P("The importance pattern explains much of the comparison. The useful information is short-memory own-volatility (which "
  "ARIMA already exploits) plus trading activity (which ARIMA cannot use). The activity features give the ML models an edge "
  "on days following unusual trading, such as the April 2025 sell-off, but on ordinary days that edge is small. "
  "Permutation importance measures predictive reliance, not causation: the number of trades does not *cause* volatility; "
  "both respond to the arrival of news.")
H2("6.8 Robustness check: the non-voting share (COMB.X0000)")
TABLE("Table 22. Identical pipeline applied to COMB.X0000",
      ["Model", "Approach", "Val QLIKE", "Test QLIKE", "Test RMSE", "Val rank", "Test rank"],
      [[i, r.approach, f(r["val QLIKE"]), f(r["test QLIKE"]), f(r["test RMSE"]), int(r["val rank (QLIKE)"]), int(r["test rank (QLIKE)"])]
       for i, r in robx.iterrows()], widths=[2.4, 1.4, 0.9, 0.9, 0.9, 0.7, 0.7], small=True)
P(f"The same specifications were selected independently (ARIMA(1,0,1), GARCH(1,1)-t; Ridge best on validation). On the "
  f"test period all five models again beat both baselines, but the winner is {SUM['X_best_test']} "
  f"({robx.loc['Random forest', 'test QLIKE']:.3f}), with Ridge second ({robx.loc['Ridge regression', 'test QLIKE']:.3f}). "
  "The X share had no comparable extreme day in the test period, so its results resemble the 'excluding 1 December' results "
  "for the voting share: models within a narrow band, baselines clearly worse.")

# ---------------------------------------------------------------- discussion
H1("7. Discussion")
H2("7.1 Best-performing model and comparison with the baselines")
P(f"On the pre-specified criterion – test QLIKE on all days – **Ridge regression is the best model** "
  f"({q(best_test):.3f}), and it was also the model chosen on validation, so validation and test agreed on the winner. It "
  f"reduced QLIKE by {impr['Ridge regression']:.0f}% relative to the 22-day historical mean and by "
  f"{imp_tab.loc['Ridge regression', 'QLIKE improvement vs naive (%)']:.0f}% relative to the naive forecast. Hypothesis H1 "
  f"is supported: every fitted model beat both baselines on the test period, by {impr[nonbase].min():.0f}–{impr[nonbase].max():.0f}% "
  "versus the historical mean. However, the *best observed* model is not the same as the model with evidence of *better "
  "generalisation*: the differences among the five fitted models are not statistically significant, and the ranking "
  "changes with one observation or with the share class.")
H2("7.2 Traditional versus machine learning – why so similar?")
P("Hypothesis H2 is supported: machine learning did not clearly outperform. The evidence points to four reasons.")
NUM(["**The signal is mostly linear and short-memory.** ARIMA(1,0,1) leaves no residual autocorrelation, the best ML model "
     "is a nearly unpenalised *linear* regression, and permutation importance is concentrated on the last two days of "
     "volatility. There is little non-linear structure for trees to find.",
     "**Small sample, high noise → overfitting.** With about 250 training days, Random Forest and Gradient Boosting fitted "
     "the training data roughly twice as well as they forecast; the gains from flexibility were eaten by variance.",
     "**Extra information helps only sometimes.** Volume and trade counts gave Ridge and Gradient Boosting an advantage in "
     "the tariff-shock window and in the high-volatility regime, which is why Ridge edges out ARIMA overall, but on ordinary "
     "days the activity variables add little.",
     "**The volatility proxy matters more than the model family.** GARCH is the only model that learns from squared "
     "close-to-close returns, which showed no detectable ARCH effects in the training data. Its parameters were imprecise, it "
     "had the worst validation QLIKE of the fitted models and it was weakest in the high-volatility regime. Its good test "
     "rank with all days comes largely from forecasting higher volatility on average, which happened to soften the 1 December "
     "miss; without that day it ranks fifth. By contrast, ARIMA on log range volatility – a traditional model using a better "
     "measurement – was as good as any ML model."])
H2("7.3 Regimes, shocks and the limits of forecasting")
P("Hypothesis H3 is only partly supported. The ordering of baselines changes sharply between regimes, but the ordering of "
  "the five fitted models is broadly similar in calm and turbulent periods. The more important regime finding is about "
  "*when* the models fail: all of them extrapolate recent history, so the costliest error is a shock that arrives when "
  "recent volatility is low (1 December 2025). No model in this study can anticipate such an event from price and volume "
  "history alone.")
H2("7.4 Financial and business implications")
BUL(["**Use a model, not the 22-day historical volatility.** For a CSE risk desk, any of the fitted models would have cut "
     "the QLIKE loss of daily volatility forecasts by roughly 30–47% relative to the simple historical average, and they "
     "react within a day to stress, while the historical average lags by weeks and then stays too high.",
     "**A simple, transparent model is sufficient.** ARIMA on log range volatility or a linear HAR-type regression with "
     "trading activity performs as well as tree ensembles, is easier to explain to a risk committee or regulator, and is "
     "less likely to break when market conditions change.",
     "**Measure volatility from the daily range.** The high–low range contains far more information about daily risk "
     "than closing prices alone; GARCH on close-to-close returns was the weakest fitted model.",
     "**Complement models with event overlays.** Forecasts systematically under-predicted the first day of a sudden shock. "
     "Scenario add-ons for scheduled or obvious events (policy announcements, natural disasters, rights issues) and stress "
     "tests are still needed.",
     "**Accuracy is not profit.** These are statistical risk forecasts. They do not predict price direction and do not, "
     "by themselves, imply any profitable trading strategy."])
H2("7.5 What cannot be concluded")
BUL(["That machine learning is generally better or worse than traditional models: the evidence covers one stock, two years and one horizon.",
     "That any model will keep its rank in 2026: rankings changed across evaluation subsets.",
     "That trading activity *causes* volatility: importance is predictive association.",
     "That the 1 December 2025 low reflects a genuine market-wide price: the data cannot tell us."])

# ---------------------------------------------------------------- limitations
H1("8. Limitations and Future Improvements")
TABLE("Table 23. Limitations and realistic improvements", ["Limitation", "Why it matters", "Improvement"],
      [["Short sample (478 days; ~250 training days)", "Too few observations for stable tree ensembles; DM tests have low power.",
        "Extend to 2015–2025 CSE data (including the 2022 crisis) and use rolling-origin evaluation over many test windows."],
       ["Single stock and single horizon", "Results may not transfer to other banks or longer horizons.",
        "Repeat on other S&P SL20 banks (HNB, SAMP, NTB) and compare 1-, 5- and 22-day horizons."],
       ["Volatility proxy ignores overnight moves; tick-size discreteness", "Range understates true daily variation; quiet days look discrete.",
        "Add the close-to-previous-close gap (Yang–Zhang-type estimator) once a reliable open price is available; use intraday data for realised volatility."],
       ["Open price unusable in 2025", "Prevents better range estimators and overnight/intraday decomposition.",
        "Obtain verified OHLC data directly from the CSE or a data vendor."],
       ["One extreme, unverified observation dominates test metrics", "Rankings and significance depend on one day.",
        "Verify against CSE trade-level records; report robust losses and multiple test windows."],
       ["Fixed parameters during the test", "Models cannot adapt to new regimes (e.g. calmer late 2025).",
        "Re-estimate on an expanding window monthly; evaluate whether adaptivity reduces over-prediction."],
       ["Only own-stock information", "Market-wide shocks (tariffs, cyclone) are not in the features.",
        "Add lagged ASPI/S&P SL20 range volatility, USD/LKR moves and an event-day dummy known in advance."],
       ["GARCH mean equation ignores return autocorrelation", "Residual autocorrelation slightly distorts the variance estimates.",
        "Use an AR(1)-GARCH or a HAR/realised-GARCH that uses the range measure directly."],
       ["FY2023 dividend amount not fully verified", "Affects one return in the training period.",
        "Confirm with the 2023 annual report/CSE announcement."]],
      widths=[2, 2.3, 3], small=True)

# ---------------------------------------------------------------- conclusion
H1("9. Conclusion")
P(f"The research question asked how effectively next-day volatility of COMB voting shares can be forecast by traditional "
  f"time-series methods compared with machine learning, and what the differences teach us. The answer is: **effectively, and "
  f"about equally well.** Daily range volatility is strongly persistent, so every model we built beat the naive and "
  f"historical-mean baselines on an untouched five-month test period, by {impr[nonbase].min():.0f}–{impr[nonbase].max():.0f}% "
  f"in QLIKE relative to the historical mean. Ridge regression on lagged volatility and trading-activity features was the best "
  f"model on both validation and test, but its advantage over ARIMA on log-volatility, Gradient Boosting, Random Forest and "
  f"GARCH was not statistically significant and reversed under reasonable sensitivity checks. The differences teach three "
  f"things: the predictable component of volatility is mainly linear and short-memory, so model flexibility adds little "
  f"and over-fits with two years of data; the measurement of volatility (range-based versus squared returns) matters more "
  f"than the model family; and all models share the same blind spot – sudden shocks that arrive in calm periods. For "
  f"practical risk management on the CSE, a transparent model on range-based volatility, combined with judgement for "
  f"identifiable events, is the defensible choice.")

# ---------------------------------------------------------------- references
H1("References")
refs = [
    "Bollerslev, T. (1986) 'Generalized autoregressive conditional heteroskedasticity', *Journal of Econometrics*, 31(3), pp. 307–327.",
    "Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.",
    "Colombo Stock Exchange (2025) *Commercial Bank of Ceylon PLC – First and final dividend for the financial year ended 31 December 2024* [CSE announcement]. Available at: https://cdn.cse.lk/cmt/announcement_portal_prod/ (Accessed: 21 October 2026).",
    "Corsi, F. (2009) 'A simple approximate long-memory model of realized volatility', *Journal of Financial Econometrics*, 7(2), pp. 174–196.",
    "Daily Mirror (2024) 'ComBank stock takes a tumble following Rs.22.5bn rights issue announcement', *Daily Mirror* (Sri Lanka), 30 May. Available at: https://www.dailymirror.lk/print/business-news/ComBank-stock-takes-a-tumble-following-Rs-22-5bn-rights-issue-announcement/273-283652 (Accessed: 21 October 2026).",
    "Daily Mirror (2025) 'ASPI plunges over 700 points below 15,000 as global trade fears grip market', *Daily Mirror* (Sri Lanka), 7 April. Available at: https://www.dailymirror.lk/print/breaking-news/ASPI-plunges-over-700-Points-below-15-000-as-global-trade-fears-grip-market/108-306211 (Accessed: 21 October 2026).",
    "Dickey, D.A. and Fuller, W.A. (1979) 'Distribution of the estimators for autoregressive time series with a unit root', *Journal of the American Statistical Association*, 74(366), pp. 427–431.",
    "Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy', *Journal of Business & Economic Statistics*, 13(3), pp. 253–263.",
    "Duan, N. (1983) 'Smearing estimate: a nonparametric retransformation method', *Journal of the American Statistical Association*, 78(383), pp. 605–610.",
    "Engle, R.F. (1982) 'Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation', *Econometrica*, 50(4), pp. 987–1007.",
    "First Capital (2024) *COMB – Earnings update: Maintain BUY* [equity research report], 4 June. Available at: https://firstcapital.lk/wp-content/uploads/2024/06/COMB-Earnings-Update-MAINTAIN-BUY-04-06-24.pdf (Accessed: 21 October 2026).",
    "Friedman, J.H. (2001) 'Greedy function approximation: a gradient boosting machine', *Annals of Statistics*, 29(5), pp. 1189–1232.",
    "Glosten, L.R., Jagannathan, R. and Runkle, D.E. (1993) 'On the relation between the expected value and the volatility of the nominal excess return on stocks', *Journal of Finance*, 48(5), pp. 1779–1801.",
    "Hansen, P.R. and Lunde, A. (2005) 'A forecast comparison of volatility models: does anything beat a GARCH(1,1)?', *Journal of Applied Econometrics*, 20(7), pp. 873–889.",
    "Harvey, D., Leybourne, S. and Newbold, P. (1997) 'Testing the equality of prediction mean squared errors', *International Journal of Forecasting*, 13(2), pp. 281–291.",
    "Hoerl, A.E. and Kennard, R.W. (1970) 'Ridge regression: biased estimation for nonorthogonal problems', *Technometrics*, 12(1), pp. 55–67.",
    "Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992) 'Testing the null hypothesis of stationarity against the alternative of a unit root', *Journal of Econometrics*, 54(1–3), pp. 159–178.",
    "Ljung, G.M. and Box, G.E.P. (1978) 'On a measure of lack of fit in time series models', *Biometrika*, 65(2), pp. 297–303.",
    "Nelson, D.B. (1991) 'Conditional heteroskedasticity in asset returns: a new approach', *Econometrica*, 59(2), pp. 347–370.",
    "Parkinson, M. (1980) 'The extreme value method for estimating the variance of the rate of return', *Journal of Business*, 53(1), pp. 61–65.",
    "Patton, A.J. (2011) 'Volatility forecast comparison using imperfect volatility proxies', *Journal of Econometrics*, 160(1), pp. 246–256.",
    "Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.",
    "Seabold, S. and Perktold, J. (2010) 'statsmodels: econometric and statistical modeling with Python', *Proceedings of the 9th Python in Science Conference*, pp. 92–96.",
    "Sheppard, K. (2024) *arch: ARCH models in Python* [software documentation]. Available at: https://arch.readthedocs.io (Accessed: 21 October 2026).",
    "The Morning (2025) [Report on the CSE and the economic fallout of Cyclone Ditwah], *The Morning* (Sri Lanka), 28 December. Available at: https://www.themorning.lk/articles/lrNxXOcESf0J9qZfm11C (Accessed: 21 October 2026).",
]
for r in refs:
    add({"t": "ref", "text": r})

# ---------------------------------------------------------------- appendices
add({"t": "pb"})
H1("Appendix A. Hyper-parameter search results (validation QLIKE)")
for k, lab in [("ridge", "Ridge regression"), ("random", "Random forest"), ("gradient", "Gradient boosting")]:
    t = tun[k]
    TABLE(f"Table A{['ridge','random','gradient'].index(k)+1}. {lab}", list(t.columns),
          [[("none" if (isinstance(v, float) and np.isnan(v)) else (f(v, 4) if c == "val_QLIKE" else (f"{v:g}" if isinstance(v, float) else v)))
            for c, v in r.items()] for _, r in t.iterrows()], small=True, highlight=[0])
TABLE("Table A4. ARIMA order search on the training sample (10 best by AIC)", ["Order", "AIC", "BIC", "Converged"],
      [[r.order, f(r.AIC, 2), f(r.BIC, 2), r.converged] for r in arsel.itertuples()], small=True, highlight=[0])
TABLE("Table A5. Ridge standardised coefficients (final model, ln-volatility scale)", ["Feature", "Coefficient", "Description"],
      [[r.feature, f(r._2, 4), r.description] for r in rcoef.itertuples()], widths=[1.3, 1, 4.5], small=True)
P("Coefficients of correlated lag features should not be interpreted individually; the negative coefficient on log "
  "volume next to a positive coefficient on the number of trades, for example, means that *many small trades* (rather than "
  "a few large block trades) signal higher next-day volatility.")

H1("Appendix B. Implementation and reproducibility")
BUL(["Code: `src/` modules (data_processing, feature_engineering, exploratory_analysis, modelling, evaluation, "
     "diagnostics, additional_analysis, pipeline) and `main.py`, which regenerates every table and figure.",
     "Notebook: `notebooks/financial_time_series_analysis.ipynb` runs the same functions step by step with explanations; "
     "it was executed end to end without errors.",
     "Outputs: 16 figures (PNG, 200 dpi), 30 CSV tables plus `all_tables.xlsx`, aligned predictions with dates "
     "(`outputs/predictions/`), and `summary.json` with the selected models.",
     "Leakage test (`check_leakage.py`): all data from 15 September 2025 onward were deliberately corrupted; every "
     "feature and every ARIMA and GARCH forecast before that date stayed identical. (The arch package initialises the "
     "first GARCH variance with a whole-sample statistic; this affects only the first days of January 2024 and is exactly "
     "zero on every validation and test date.)",
     "Determinism: Random Forest and Gradient Boosting use random_state = 42; all other steps are deterministic.",
     "Software: Python 3.13, pandas, NumPy, SciPy, statsmodels 0.15, arch 8.0, scikit-learn 1.9, matplotlib, seaborn."])

H1("Appendix C. Group presentation preparation")
H2("C.1 Talking points (one per slide)")
NUM([f"**Problem:** forecasting tomorrow's risk of COMB shares, used for VaR, margins and position limits on the CSE.",
     "**Why volatility:** prices are non-stationary and returns are mostly noise; ln range-volatility is stationary and strongly autocorrelated.",
     "**Data:** 478 trading days, 2024–2025; three securities in one file; open price unusable in 2025; dividends and a rights issue corrected.",
     "**Target:** Parkinson high–low volatility, forecast one day ahead from the close of the previous day.",
     f"**Split:** train Feb 2024–Feb 2025 ({tr_n}), validation Mar–Jul 2025 ({va_n}), test Aug–Dec 2025 ({te_n}); test used once.",
     "**Models and why:** naive and 22-day mean (benchmarks); ARIMA(1,0,1) on ln-vol (stationary, ARMA-like ACF); GARCH(1,1)-t "
     "(classic clustering model, fat tails); Ridge (correlated features, small data); Random Forest and Gradient Boosting (non-linearity).",
     f"**Headline result:** Ridge best on validation and test (test QLIKE {q(best_test):.3f}); every model "
     f"{impr[nonbase].min():.0f}–{impr[nonbase].max():.0f}% better than the 22-day mean; differences among models not significant.",
     "**Additional investigation:** regimes – advantage over baselines holds in calm and turbulent periods; biggest miss = shock in a calm period (1 Dec 2025).",
     "**Interpretation:** signal is linear and short-memory; trees overfit; the range-based measure beats close-to-close GARCH.",
     "**Limitations:** short sample, one stock, one extreme unverified day, no market-wide features."])
H2("C.2 Likely lecturer questions and evidence-based answers")
qa_list = [
    ("Why did you not forecast the share price or return?",
     "Log prices have a unit root (ADF p = 0.946), so price forecasts are dominated by the random walk; return magnitudes are close to unpredictable. Ln-volatility is stationary and has Ljung–Box Q(10) p < 0.001, so it is a well-posed forecasting target with a clear risk-management use."),
    ("Why the Parkinson estimator?",
     "It uses the full intraday range, is far more efficient than a squared return, is unaffected by dividends/rights, and needs only high/low – the open price is unusable in 2025."),
    ("How do you know there is no data leakage?",
     "Every feature is shifted one day; rolling windows end at day t; scaler inside a Pipeline fitted on training rows; ARIMA order, regime threshold and stationarity tests use training data; hyper-parameters chosen on validation; the test period was used once after everything was fixed."),
    ("Why is test QLIKE so much higher than validation?",
     "One day (1 Dec 2025, actual 11.2%) dominates. Without it, test QLIKE is 0.50–0.55 for the fitted models, similar to validation."),
    ("Isn't excluding the outlier cheating?",
     "We did not exclude it from the main results; we report both. The sensitivity analysis shows the ranking is fragile, which is a finding in itself."),
    ("Why does Ridge beat Random Forest if RF is more powerful?",
     "RF fitted training data about twice as well as it forecast (train QLIKE 0.36 vs validation 0.68): overfitting with ~250 observations. The relationships are mostly linear."),
    ("Why did GARCH do relatively poorly on validation?",
     "Squared daily returns showed no significant ARCH effects (ARCH-LM p = 0.36), so GARCH's dynamics were estimated imprecisely (β p ≈ 0.07, persistence 0.73). It uses a noisier volatility proxy than the other models."),
    ("Is the best model significantly better?",
     "No. Diebold–Mariano tests against Ridge give p > 0.2 for every other model, including the baselines, because one extreme day inflates the variance of the loss differences. We distinguish best observed performance from evidence of better generalisation."),
    ("Can this make money?",
     "Not directly. These are risk forecasts, not direction forecasts; accuracy in QLIKE does not imply a profitable strategy."),
    ("What would you do with more time?",
     "Longer history and several banks, rolling-origin evaluation, market-wide predictors (ASPI volatility, USD/LKR), and verified OHLC data to use better range estimators."),
]
for qq, aa_ in qa_list:
    P(f"**Q: {qq}**  {aa_}")
H2("C.3 Suggested division of work (with deliberate overlap)")
TABLE("Table C1. Roles", ["Member", "Lead responsibility", "Must also understand / review"],
      [["Member 1", "Data inspection, cleaning, corporate actions, EDA and statistical tests (Sections 3–4)", "Feature engineering; why the target was chosen"],
       ["Member 2", "Baselines, ARIMA and GARCH models and their diagnostics (Sections 5.2–5.4, 6.4)", "Evaluation metrics; DM tests"],
       ["Member 3", "Feature engineering, ML models, tuning, permutation importance, regime analysis (5.5–5.6, 6.5–6.7)", "Leakage prevention; ARIMA results"],
       ["Member 4", "Evaluation framework, comparison tables, robustness check, report integration and slides (6.1–6.3, 6.8, 7–9)", "Every model's selection rationale"]],
      widths=[1, 3.5, 2.5])
P("Every member should be able to explain the research question, the chronological split, the baseline logic and the headline "
  "result without notes, and should run `python main.py` once on their own computer.")

# static contents list (Word TOC fields are not filled by LibreOffice PDF export)
heads = [b for b in B if b["t"] in ("h1", "h2")]
toc = [{"t": "h1", "text": "Contents"}] + [{"t": "toc1" if h["t"] == "h1" else "toc2", "text": h["text"]} for h in heads]
i = next(k for k, b in enumerate(B) if b["t"] == "TOCPLACE")
B[i:i + 1] = toc
json.dump({"title": "Forecasting Next-Day Volatility of COMB", "blocks": B}, open(HERE / "report_content.json", "w"), indent=1)
print("blocks:", len(B))
