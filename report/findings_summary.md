# Findings summary – COMB next-day volatility forecasting

**Data.** 478 CSE trading days (2 Jan 2024 – 31 Dec 2025) for COMB.N0000. The file also had the non-voting share and
rights entitlements, which were separated. The open price is unusable in 2025. Returns were adjusted for two dividends
and the July 2024 rights issue.

**Target.** Next-day Parkinson (high–low) volatility. Log prices are non-stationary (ADF p = 0.946). Ln-volatility is
stationary and strongly autocorrelated (Ljung–Box Q(10) = 130.6, p < 0.001). Squared returns show no detectable ARCH
effects (ARCH-LM p = 0.364).

**Models.** Baselines: naive and 22-day mean. Traditional: ARIMA(1,0,1) on ln-vol (AIC choice) and GARCH(1,1)-t
(chosen from three variants on validation). ML: Ridge (α = 0.01), Random Forest (depth 6), Gradient Boosting
(100 trees, depth 2).

**Main results (test, Aug–Dec 2025, 102 days, QLIKE).**
- Ridge regression was best: 1.658 on test and 0.587 on validation, so validation and test picked the same winner.
- Every model beat both baselines, by 31–47% compared with the 22-day mean (3.140).
- Differences among the five models were not statistically significant (Diebold–Mariano p > 0.2 against Ridge).
- One day, 1 Dec 2025 (actual 11.2%, after Cyclone Ditwah), dominates the test losses. Without it the ranking is
  ARIMA 0.504, RF 0.505, GBM 0.512, Ridge 0.515, GARCH 0.548. On the non-voting share, Random Forest came first.
- The tree models overfit: training QLIKE was about 0.33–0.36 against about 0.66–0.68 on validation.

**Additional investigation (regimes).**
- The models' advantage over the baselines held in both calm and turbulent regimes.
- The naive forecast failed in calm periods and the historical mean failed in turbulent ones.
- The costliest errors came from a shock that arrived in a calm regime.
- Permutation importance shows that the forecasts rely on the last 1–2 days of volatility, the number of trades and
  recent returns.

**Interpretation.** The forecastable part of volatility is mostly linear and short-memory, so ML adds little over a
well-specified ARIMA. Measuring volatility from the daily range matters more than which model family is used.
These are risk forecasts. They do not show a profitable trading strategy.
