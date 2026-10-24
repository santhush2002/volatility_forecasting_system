"""
config.py
---------
One place for every setting used in the project, so that the whole analysis
can be re-run with identical choices (reproducibility).

If you change something here (for example the split dates), re-run main.py
and every table, figure and number in outputs/ will be regenerated.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root, so the project works on any computer)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "original" / "COMB_Stock_Data_2024_2025.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
PRED_DIR = ROOT / "outputs" / "predictions"
MODEL_DIR = ROOT / "outputs" / "model_results"

for _d in (PROCESSED_DIR, FIG_DIR, TAB_DIR, PRED_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Which security we model
# ---------------------------------------------------------------------------
# The Excel file mixes THREE securities of Commercial Bank of Ceylon PLC:
#   N = ordinary voting share      (COMB.N0000)  <- main analysis
#   X = ordinary non-voting share  (COMB.X0000)  <- robustness check
#   R = rights entitlements traded 30 Jul - 6 Aug 2024 (excluded)
MAIN_CLASS = "N"
ROBUST_CLASS = "X"

# ---------------------------------------------------------------------------
# Corporate actions inside the sample.
# Unadjusted closing prices fall mechanically on these days, which is NOT a
# market movement. We correct close-to-close returns on these dates.
#   type "dividend": value = total dividend per share (cash + scrip), Rs.
#   type "rights":   ratio = new shares per old share, price = subscription
# Sources are documented in the report (CSE announcement / news articles).
# ---------------------------------------------------------------------------
CORPORATE_ACTIONS = {
    "N": [
        {"date": "2024-04-01", "type": "dividend", "value": 6.50,
         "note": "FY2023 final dividend (DPS Rs 6.50; cash/scrip split not verified)"},
        {"date": "2024-07-16", "type": "rights", "ratio": 1 / 5, "price": 85.0,
         "note": "1-for-5 rights issue at Rs 85 (ex-rights date inferred from data)"},
        {"date": "2025-04-01", "type": "dividend", "value": 9.50,
         "note": "FY2024 dividend Rs 7.50 cash + Rs 2.00 scrip, XD 1 Apr 2025 (CSE)"},
    ],
    "X": [
        {"date": "2024-04-01", "type": "dividend", "value": 6.50,
         "note": "FY2023 final dividend (assumed same DPS as voting share)"},
        {"date": "2024-07-16", "type": "rights", "ratio": 1 / 5, "price": 68.0,
         "note": "1-for-5 rights issue at Rs 68 (ex-rights date inferred from data)"},
        {"date": "2025-04-01", "type": "dividend", "value": 9.50,
         "note": "FY2024 dividend Rs 7.50 cash + Rs 2.00 scrip, XD 1 Apr 2025 (CSE)"},
    ],
}

# ---------------------------------------------------------------------------
# Chronological split (dates are the FORECAST TARGET dates)
#   train      : 2024 start  -> 28 Feb 2025   (parameter estimation)
#   validation : 1 Mar 2025  -> 31 Jul 2025   (model / hyper-parameter choice)
#   test       : 1 Aug 2025  -> 31 Dec 2025   (untouched final comparison)
# ---------------------------------------------------------------------------
TRAIN_END = "2025-02-28"
VAL_END = "2025-07-31"

# Rolling windows used for features / baselines (trading days)
WEEK = 5
MONTH = 22

RANDOM_STATE = 42
