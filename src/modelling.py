"""
modelling.py
------------
Stages 9-11 - baselines, traditional time-series models and machine-learning
models, all producing ONE-STEP-AHEAD forecasts of next-day volatility.

Common protocol for every model (this is what makes the comparison fair):
  * Same target  : Parkinson volatility of day t+1 (percent).
  * Same origin  : forecast made at the close of day t with data up to t.
  * Same dates   : identical validation and test target dates.
  * Validation   : parameters estimated on TRAIN only; forecasts produced for
                   the validation period with those fixed parameters; the
                   model specification / hyper-parameters with the lowest
                   validation QLIKE is chosen.
  * Test         : the chosen specification is re-estimated ONCE on
                   TRAIN + VALIDATION, then produces forecasts for the test
                   period with fixed parameters (new observations are fed in
                   as they arrive, but nothing is re-tuned).
  * Back-transform: models fitted on log-volatility are converted back with
                   Duan's smearing factor estimated on the fitting sample.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from . import config
from .evaluation import qlike
from .feature_engineering import FEATURES


def smearing_factor(log_y, log_fit) -> float:
    """Duan (1983) smearing: E[exp(error)] so that exp(log forecast) is not
    biased downwards when converted back to the volatility scale."""
    return float(np.mean(np.exp(np.asarray(log_y) - np.asarray(log_fit))))


# ===========================================================================
# 1. BASELINES (defined before seeing any results)
# ===========================================================================
BASELINES = {
    "Naive (sigma_t)": "naive",                 # tomorrow = today
    "Historical mean (22-day)": "hist_mean22",  # tomorrow = last month's average
}


# ===========================================================================
# 2. ARIMA on log volatility
# ===========================================================================
def _arima_one_step(log_vol: pd.Series, fit_end: str, order, alpha=0.2):
    """Fit ARIMA on log_vol up to fit_end, then produce one-step-ahead
    forecasts for EVERY date using the fixed fitted parameters."""
    y_fit = log_vol.loc[:fit_end].values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = ARIMA(y_fit, order=order, trend="c" if order[1] == 0 else "n").fit()
        conv_warn = [str(x.message) for x in w if "converge" in str(x.message).lower()]
    full = res.apply(log_vol.values, refit=False)       # same parameters, all data
    pred = full.get_prediction()                        # one-step-ahead predictions
    mean = pd.Series(pred.predicted_mean, index=log_vol.index)
    ci = pred.conf_int(alpha=alpha)
    lo = pd.Series(ci[:, 0], index=log_vol.index)
    hi = pd.Series(ci[:, 1], index=log_vol.index)
    return res, mean, lo, hi, conv_warn


def select_arima(log_vol: pd.Series, train_idx, max_p=3, max_q=3, d=0):
    """Grid-search ARIMA(p,d,q) by AIC on the TRAINING sample only."""
    rows = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(log_vol.loc[:config.TRAIN_END].values, order=(p, d, q),
                            trend="c" if d == 0 else "n").fit()
            rows.append({"order": (p, d, q), "AIC": res.aic, "BIC": res.bic,
                         "converged": res.mle_retvals.get("converged", True)})
        except Exception as e:  # pragma: no cover
            rows.append({"order": (p, d, q), "AIC": np.nan, "BIC": np.nan, "converged": False})
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def run_arima(g, data, train, val, test, order):
    """Validation forecasts (fit on train) and test forecasts (fit on train+val)."""
    log_vol = g["log_vol"]
    out = {}
    for stage, fit_end, idx in [("val", config.TRAIN_END, train.index.append(val.index)),
                                ("test", config.VAL_END, data.index)]:
        res, mean, lo, hi, warn = _arima_one_step(log_vol, fit_end, order)
        fit_rows = data.loc[:fit_end].index
        s = smearing_factor(np.log(data.loc[fit_rows, "y"]), mean.loc[fit_rows])
        out[stage] = {"res": res, "warnings": warn, "smear": s,
                      "pred": np.exp(mean) * s, "lo": np.exp(lo), "hi": np.exp(hi),
                      "log_mean": mean}
    return out


# ===========================================================================
# 3. GARCH family on adjusted returns
# ===========================================================================
GARCH_SPECS = {
    "GARCH(1,1)-t": dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
    "GJR-GARCH(1,1)-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH(1,1)-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}


def _garch_cond_vol(ret: pd.Series, fit_end: str, spec: dict):
    """Estimate on returns up to fit_end; then compute the conditional
    volatility for all days with the SAME parameters. arch's conditional
    volatility on day t is based only on returns up to t-1, i.e. it is a
    genuine one-step-ahead forecast for day t."""
    r = ret.dropna()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = arch_model(r.loc[:fit_end], mean="Constant", **spec).fit(disp="off")
        conv_warn = [str(x.message) for x in w]
    fixed = arch_model(r, mean="Constant", **spec).fix(res.params)
    return res, pd.Series(fixed.conditional_volatility, index=r.index), conv_warn


def run_garch(g, data, spec_name):
    """GARCH volatility is close-to-close; our target is intraday range
    volatility. We therefore map GARCH sigma onto the target scale with a
    linear regression on logs estimated on the FITTING sample only:
        ln(y_t) = a + b * ln(garch_sigma_t)
    """
    spec = GARCH_SPECS[spec_name]
    out = {}
    for stage, fit_end in [("val", config.TRAIN_END), ("test", config.VAL_END)]:
        res, cv, warn = _garch_cond_vol(g["ret"], fit_end, spec)
        cv = cv.reindex(data.index)
        fit_rows = data.loc[:fit_end].index
        X = np.log(cv.loc[fit_rows].values).reshape(-1, 1)
        lr = LinearRegression().fit(X, np.log(data.loc[fit_rows, "y"].values))
        log_pred = pd.Series(lr.predict(np.log(cv.values).reshape(-1, 1)), index=data.index)
        s = smearing_factor(np.log(data.loc[fit_rows, "y"]), log_pred.loc[fit_rows])
        out[stage] = {"res": res, "warnings": warn, "calib_a": float(lr.intercept_),
                      "calib_b": float(lr.coef_[0]), "smear": s,
                      "pred": np.exp(log_pred) * s, "garch_sigma": cv}
    return out


# ===========================================================================
# 4. MACHINE LEARNING (Ridge, Random Forest, Gradient Boosting)
# ===========================================================================
ML_GRIDS = {
    "Ridge regression": {
        "make": lambda p: Pipeline([("scale", StandardScaler()),   # scaler fitted on training rows only
                                    ("model", Ridge(alpha=p["alpha"]))]),
        "grid": [{"alpha": a} for a in [0.01, 0.1, 1, 10, 30, 100, 300]],
    },
    "Random forest": {
        "make": lambda p: RandomForestRegressor(n_estimators=500, random_state=config.RANDOM_STATE,
                                                n_jobs=-1, **p),
        "grid": [{"max_depth": d, "min_samples_leaf": l, "max_features": f}
                 for d in [3, 6, None] for l in [5, 15] for f in [0.33, 1.0]],
    },
    "Gradient boosting": {
        "make": lambda p: GradientBoostingRegressor(random_state=config.RANDOM_STATE, subsample=0.8,
                                                    loss="squared_error", **p),
        "grid": [{"learning_rate": lr, "n_estimators": n, "max_depth": d}
                 for lr in [0.02, 0.05] for n in [100, 300] for d in [2, 3]],
    },
}


def tune_ml(name, train, val):
    """Fit every grid point on TRAIN, score on VALIDATION (QLIKE).
    Returns the full tuning table and the best parameter set."""
    spec = ML_GRIDS[name]
    rows = []
    for p in spec["grid"]:
        model = spec["make"](p).fit(train[FEATURES], train["log_y"])
        s = smearing_factor(train["log_y"], model.predict(train[FEATURES]))
        f = np.exp(model.predict(val[FEATURES])) * s
        rows.append({**p, "val_QLIKE": qlike(val["y"].values, f)})
    tab = pd.DataFrame(rows).sort_values("val_QLIKE").reset_index(drop=True)
    best = {k: tab.loc[0, k] for k in spec["grid"][0].keys()}
    # pandas may turn None into NaN and ints into floats - undo that
    for k, v in best.items():
        if isinstance(v, float) and np.isnan(v):
            best[k] = None
        elif k in ("max_depth", "min_samples_leaf", "n_estimators") and v is not None:
            best[k] = int(v)
    return tab, best


def run_ml(name, params, data, train, val, test):
    spec = ML_GRIDS[name]
    out = {}
    for stage, fit in [("val", train), ("test", pd.concat([train, val]))]:
        model = spec["make"](params).fit(fit[FEATURES], fit["log_y"])
        s = smearing_factor(fit["log_y"], model.predict(fit[FEATURES]))
        pred = pd.Series(np.exp(model.predict(data[FEATURES])) * s, index=data.index)
        out[stage] = {"model": model, "smear": s, "pred": pred}
    return out
