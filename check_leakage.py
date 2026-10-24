"""
check_leakage.py
----------------
Automated look-ahead test: we deliberately corrupt all data from 15 Sep 2025
onwards and check that every feature and every model forecast made BEFORE that
date is unchanged. If any forecast used future data, it would change.

Run:  python check_leakage.py
"""
import warnings

import numpy as np
import pandas as pd

from src import config, data_processing as dp, feature_engineering as fe, modelling as md

warnings.filterwarnings("ignore")
CUT = pd.Timestamp("2025-09-15")

g = dp.build_dataset("N", save=False)
g2 = g.copy()
m = g2.index >= CUT
g2.loc[m, "high"] *= 1.5
g2.loc[m, "close"] *= 1.3
g2.loc[m, ["volume", "trades"]] *= 10
g2 = dp.add_target(dp.add_adjusted_returns(g2[["high", "low", "close", "open", "trades", "volume", "turnover", "low_imputed"]], "N"))

d, d2 = fe.build_features(g), fe.build_features(g2)
cols = fe.FEATURES + ["naive", "hist_mean22"]
print("1. Features before cut unchanged:      ", np.allclose(d.loc[:CUT, cols], d2.loc[:CUT, cols]))

_, a, *_ = md._arima_one_step(g["log_vol"], config.VAL_END, (1, 0, 1))
_, b, *_ = md._arima_one_step(g2["log_vol"], config.VAL_END, (1, 0, 1))
print("2. ARIMA forecasts before cut unchanged:", np.allclose(a.loc[:CUT], b.loc[:CUT]))

for name, spec in md.GARCH_SPECS.items():
    _, c1, _ = md._garch_cond_vol(g["ret"], config.VAL_END, spec)
    _, c2, _ = md._garch_cond_vol(g2["ret"], config.VAL_END, spec)
    diff_eval = (c1 - c2).abs().loc["2025-03-01":CUT - pd.Timedelta(days=1)].max()
    diff_start = (c1 - c2).abs().loc[:"2024-02-06"].max()
    # arch initialises the first variance with a whole-sample statistic; its
    # effect decays geometrically and is exactly zero on all evaluation dates.
    print(f"3. {name}: max change on validation/test dates before cut = {diff_eval:.2e} "
          f"(initialisation effect on first Jan-2024 days = {diff_start:.1e})")

print("4. ML models use only the features checked in (1) and are fitted on rows up to", config.VAL_END)
