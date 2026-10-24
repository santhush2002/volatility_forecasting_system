"""
feature_engineering.py
----------------------
Stage 7 - building predictors WITHOUT data leakage.

Golden rule used here: every row is a forecast for trading day t+1 that is
made at the close of trading day t. So every feature may only use data from
day t or earlier. We achieve that by computing each feature on day t and then
moving it forward one row with .shift(1) so it sits next to the day-(t+1)
target.

Rows are indexed by the TARGET date (the day being forecast). This makes the
train/validation/test split by date unambiguous.
"""
import numpy as np
import pandas as pd

from . import config

FEATURE_DESCRIPTIONS = {
    "lv_0": "log Parkinson volatility on day t (most recent day)",
    "lv_1": "log Parkinson volatility on day t-1",
    "lv_2": "log Parkinson volatility on day t-2",
    "lv_w": "log of the 5-day average Parkinson volatility (HAR 'weekly' term)",
    "lv_m": "log of the 22-day average Parkinson volatility (HAR 'monthly' term)",
    "absret_0": "absolute adjusted return on day t (size of last price move, %)",
    "absret_w": "average absolute adjusted return over the last 5 days (%)",
    "ret_0": "signed adjusted return on day t (captures the 'leverage effect')",
    "ret_m": "cumulative adjusted return over the last 22 days (momentum / trend, %)",
    "log_volume_0": "log of shares traded on day t (activity / liquidity)",
    "abn_volume": "log(volume_t / 22-day average volume): unusual activity",
    "log_trades_0": "log of number of trades on day t",
    "days_to_next": "calendar days until the next trading day (known from the CSE holiday calendar)",
    "monday_next": "1 if the forecast day is a Monday",
}
FEATURES = list(FEATURE_DESCRIPTIONS.keys())


def build_features(g: pd.DataFrame) -> pd.DataFrame:
    """Create the modelling table: target y (next-day volatility) + features."""
    d = pd.DataFrame(index=g.index)

    # ---- quantities measured ON day t (will be shifted afterwards) --------
    pv = g["park_vol"]
    on_day_t = pd.DataFrame(index=g.index)
    on_day_t["lv_0"] = np.log(pv)
    on_day_t["lv_1"] = np.log(pv).shift(1)
    on_day_t["lv_2"] = np.log(pv).shift(2)
    # rolling windows end at day t (pandas rolling looks BACKWARD only)
    on_day_t["lv_w"] = np.log(pv.rolling(config.WEEK).mean())
    on_day_t["lv_m"] = np.log(pv.rolling(config.MONTH).mean())
    on_day_t["absret_0"] = g["ret"].abs()
    on_day_t["absret_w"] = g["ret"].abs().rolling(config.WEEK).mean()
    on_day_t["ret_0"] = g["ret"]
    on_day_t["ret_m"] = g["ret"].rolling(config.MONTH).sum()
    on_day_t["log_volume_0"] = np.log(g["volume"])
    on_day_t["abn_volume"] = np.log(g["volume"] / g["volume"].rolling(config.MONTH).mean())
    on_day_t["log_trades_0"] = np.log(g["trades"])
    # simple baselines are also "features known at t"
    on_day_t["naive"] = pv                                  # sigma_t
    on_day_t["hist_mean22"] = pv.rolling(config.MONTH).mean()  # 22-day historical vol

    # ---- shift everything forward: row t+1 now holds information of day t --
    d = on_day_t.shift(1)

    # ---- calendar information about the target day (known in advance) ----
    idx = pd.Series(g.index, index=g.index)
    d["days_to_next"] = idx.diff().dt.days      # gap between day t and day t+1
    d["monday_next"] = (g.index.dayofweek == 0).astype(int)

    # ---- the target -------------------------------------------------------
    d["y"] = pv                 # volatility ON the target day
    d["log_y"] = np.log(pv)
    d["ret_target_day"] = g["ret"]  # kept only for GARCH alignment checks

    d = d.dropna(subset=FEATURES + ["y"])
    return d


def split(d: pd.DataFrame):
    """Chronological split by TARGET date (never random)."""
    train = d.loc[:config.TRAIN_END]
    val = d.loc[pd.Timestamp(config.TRAIN_END) + pd.Timedelta(days=1):config.VAL_END]
    test = d.loc[pd.Timestamp(config.VAL_END) + pd.Timedelta(days=1):]
    return train, val, test


def split_table(train, val, test) -> pd.DataFrame:
    rows = []
    n = len(train) + len(val) + len(test)
    for name, part in [("Training", train), ("Validation", val), ("Test", test)]:
        rows.append({"set": name, "first target date": part.index.min().date(),
                     "last target date": part.index.max().date(),
                     "observations": len(part), "share": f"{100 * len(part) / n:.1f}%",
                     "used for": {"Training": "estimating parameters",
                                  "Validation": "choosing orders / hyper-parameters",
                                  "Test": "final, untouched comparison"}[name]})
    return pd.DataFrame(rows)
