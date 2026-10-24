"""
data_processing.py
------------------
Stage 4 - loading, checking and cleaning the raw CSE data.

What this module does, in plain language:
1. Reads the original Excel file WITHOUT changing it (the raw copy is kept).
2. Separates the three securities that are mixed in the file (N, X, R).
3. Checks data quality: dates, sorting, duplicates, gaps, missing values,
   impossible prices (e.g. close above the day's high).
4. Repairs the few missing LOW prices using information from the SAME day only.
5. Builds corporate-action-adjusted daily log returns.
6. Builds the forecasting target: the Parkinson (high-low range) daily
   volatility.
"""
import numpy as np
import pandas as pd

from . import config

COLS = {
    "COMPANY ID": "company",
    "MAIN TYPE": "share_class",
    "SUB TYPE": "sub_type",
    "SHORT NAME": "name",
    "TRADING DATE": "date",
    "PRICE HIGH (Rs.)": "high",
    "PRICE LOW (Rs.)": "low",
    "CLOSE PRICE (Rs.)": "close",
    "OPEN PRICE (Rs.)": "open",
    "TRADE VOLUME (No.)": "trades",
    "SHARE VOLUME (No.)": "volume",
    "TURNOVER (Rs.)": "turnover",
}


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    """Read the untouched Excel file and give the columns short names."""
    raw = pd.read_excel(path)
    df = raw.rename(columns=COLS).copy()
    df["date"] = pd.to_datetime(df["date"])  # parse text dates -> real dates
    return df


def raw_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per security in the file: how many rows and which period."""
    out = (df.groupby(["share_class", "sub_type"])
             .agg(rows=("date", "size"), start=("date", "min"), end=("date", "max"))
             .reset_index())
    meaning = {"N": "Ordinary voting share (COMB.N0000)",
               "X": "Ordinary non-voting share (COMB.X0000)",
               "R": "Rights entitlement (traded 30 Jul-6 Aug 2024)"}
    out["meaning"] = out["share_class"].map(meaning)
    return out


def quality_report(df: pd.DataFrame, share_class: str) -> pd.DataFrame:
    """Data-quality summary table for one security (before any cleaning)."""
    g = df[df["share_class"] == share_class].copy()
    gaps = g["date"].diff().dt.days.dropna()
    open_eq_close_2025 = (g.loc[g["date"].dt.year == 2025, "open"]
                          == g.loc[g["date"].dt.year == 2025, "close"]).mean()
    open_eq_close_2024 = (g.loc[g["date"].dt.year == 2024, "open"]
                          == g.loc[g["date"].dt.year == 2024, "close"]).mean()
    checks = {
        "Rows (trading days)": len(g),
        "First date": g["date"].min().date(),
        "Last date": g["date"].max().date(),
        "Sorted chronologically": bool(g["date"].is_monotonic_increasing),
        "Duplicate dates": int(g["date"].duplicated().sum()),
        "Missing HIGH": int(g["high"].isna().sum()),
        "Missing LOW": int(g["low"].isna().sum()),
        "Missing OPEN": int(g["open"].isna().sum()),
        "Missing CLOSE": int(g["close"].isna().sum()),
        "Zero / negative prices": int((g[["high", "low", "close"]] <= 0).sum().sum()),
        "Close outside [Low, High]": int(((g["close"] > g["high"]) | (g["close"] < g["low"])).sum()),
        "High < Low": int((g["high"] < g["low"]).sum()),
        "Days with zero range (High = Low)": int((g["high"] == g["low"]).sum()),
        "Gap of 1 calendar day": int((gaps == 1).sum()),
        "Gap of 2-3 calendar days (weekends)": int(gaps.between(2, 3).sum()),
        "Gap of 4-5 calendar days (holidays)": int(gaps.between(4, 5).sum()),
        "Max gap (calendar days)": int(gaps.max()),
        "Share of days OPEN == CLOSE (2024)": round(open_eq_close_2024, 3),
        "Share of days OPEN == CLOSE (2025)": round(open_eq_close_2025, 3),
    }
    return pd.DataFrame({"check": list(checks.keys()), "result": [str(v) for v in checks.values()]})


def clean_class(df: pd.DataFrame, share_class: str) -> pd.DataFrame:
    """Return a clean, date-indexed table for one security.

    Decisions (all documented in the report):
    * OPEN is NOT used anywhere: from 1 Jan 2025 it equals CLOSE on 100% of
      days, i.e. the field is not a genuine opening price in this export.
    * Missing LOW (4 days, all in 2024): replaced by min(open, close) of the
      SAME day. This uses no future information. It slightly understates the
      day's range, so these days are flagged.
    * Outliers are NOT deleted: large moves (e.g. April 2025) are real events.
    """
    g = (df[df["share_class"] == share_class]
         .sort_values("date").drop_duplicates("date").set_index("date").copy())
    g = g[["high", "low", "close", "open", "trades", "volume", "turnover"]]

    g["low_imputed"] = g["low"].isna()
    same_day_floor = g[["open", "close"]].min(axis=1)  # same-day info only
    g["low"] = g["low"].fillna(same_day_floor)
    g["low"] = np.minimum(g["low"], g["close"])         # keep low <= close
    assert (g["high"] >= g["low"]).all()
    return g


def add_adjusted_returns(g: pd.DataFrame, share_class: str) -> pd.DataFrame:
    """Daily log return of the close, corrected for dividends and the rights issue.

    Ordinary day:  r_t = ln(C_t / C_{t-1})
    Ex-dividend:   r_t = ln((C_t + D) / C_{t-1})       (D = dividend per share)
    Ex-rights:     r_t = ln(C_t / TERP_{t-1}),
                   TERP = (C_{t-1} + k * S) / (1 + k)  (k = 1/5, S = subscription price)

    These corrections only use information announced before the event, so
    they create no look-ahead bias.
    """
    g = g.copy()
    prev = g["close"].shift(1)
    g["ret_raw"] = 100 * np.log(g["close"] / prev)
    adj_num, adj_den = g["close"].copy(), prev.copy()
    g["corp_action"] = ""
    for ca in config.CORPORATE_ACTIONS[share_class]:
        d = pd.Timestamp(ca["date"])
        if d not in g.index:
            continue
        if ca["type"] == "dividend":
            adj_num.loc[d] = g.loc[d, "close"] + ca["value"]
        elif ca["type"] == "rights":
            k, s = ca["ratio"], ca["price"]
            adj_den.loc[d] = (prev.loc[d] + k * s) / (1 + k)
        g.loc[d, "corp_action"] = ca["type"]
    g["ret"] = 100 * np.log(adj_num / adj_den)   # percent log return
    return g


def add_target(g: pd.DataFrame) -> pd.DataFrame:
    """Parkinson (1980) range-based daily volatility, in percent.

        sigma_t = 100 * sqrt( (ln H_t - ln L_t)^2 / (4 ln 2) )

    Why this measure: it uses the whole day's trading range, so it is a far
    less noisy estimate of daily volatility than a single squared return, and
    it is unaffected by dividends / rights (which shift the overnight price,
    not the intraday range).
    """
    g = g.copy()
    hl = np.log(g["high"] / g["low"])
    g["park_vol"] = 100 * np.sqrt(hl ** 2 / (4 * np.log(2)))
    g["log_vol"] = np.log(g["park_vol"])
    return g


def build_dataset(share_class: str = config.MAIN_CLASS, save: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline for one share class."""
    raw = load_raw()
    g = clean_class(raw, share_class)
    g = add_adjusted_returns(g, share_class)
    g = add_target(g)
    if save:
        g.to_csv(config.PROCESSED_DIR / f"COMB_{share_class}_clean.csv")
    return g


def corporate_action_table(g: pd.DataFrame, share_class: str) -> pd.DataFrame:
    rows = []
    for ca in config.CORPORATE_ACTIONS[share_class]:
        d = pd.Timestamp(ca["date"])
        rows.append({"date": d.date(), "type": ca["type"], "note": ca["note"],
                     "raw_return_%": round(g.loc[d, "ret_raw"], 3),
                     "adjusted_return_%": round(g.loc[d, "ret"], 3)})
    return pd.DataFrame(rows)
