"""
evaluation.py
-------------
Stage 13 - forecast-accuracy measures and a statistical test for comparing
two forecasts.

All metrics are computed on the SAME scale for every model: daily Parkinson
volatility in percent (sigma), or its square (variance) for QLIKE.
"""
import numpy as np
import pandas as pd
from scipy import stats


def mae(y, f):
    """Mean Absolute Error (percentage points of daily volatility). Lower = better."""
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    """Root Mean Squared Error. Penalises big misses more than MAE. Lower = better."""
    return float(np.sqrt(np.mean((y - f) ** 2)))


def qlike_series(y, f):
    """QLIKE loss for each day, computed on variances (sigma^2).

        QLIKE = y^2/f^2 - ln(y^2/f^2) - 1     (0 when f = y, always >= 0)

    Patton (2011) showed QLIKE ranks volatility forecasts correctly even when
    the 'true' volatility is only observed through a noisy proxy. It punishes
    UNDER-prediction of risk more than over-prediction, which matches what a
    risk manager cares about.
    """
    ratio = (np.asarray(y) ** 2) / (np.asarray(f) ** 2)
    return ratio - np.log(ratio) - 1


def qlike(y, f):
    return float(np.mean(qlike_series(y, f)))


def mz_r2(y, f):
    """Mincer-Zarnowitz R^2: R^2 of regressing actual on forecast.
    Measures how much of the variation in actual volatility the forecast
    tracks, ignoring bias. Higher = better."""
    y, f = np.asarray(y), np.asarray(f)
    if np.std(f) == 0:
        return np.nan
    return float(np.corrcoef(y, f)[0, 1] ** 2)


def all_metrics(y, f) -> dict:
    return {"QLIKE": qlike(y, f), "RMSE": rmse(y, f), "MAE": mae(y, f),
            "MZ_R2": mz_r2(y, f), "mean_forecast": float(np.mean(f)),
            "mean_actual": float(np.mean(y))}


def diebold_mariano(loss_a, loss_b, h: int = 1):
    """Diebold-Mariano test with the Harvey et al. (1997) small-sample correction.

    H0: models A and B have equal expected loss.
    d_t = loss_a - loss_b; a NEGATIVE statistic means A has the lower loss.
    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    d_bar = d.mean()
    # long-run variance with h-1 autocovariances (h=1 -> simple variance)
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        lrv += 2 * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
    dm = d_bar / np.sqrt(lrv / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


def metrics_table(pred: pd.DataFrame, models: list, actual_col: str = "y") -> pd.DataFrame:
    rows = []
    for m in models:
        r = all_metrics(pred[actual_col].values, pred[m].values)
        r["model"] = m
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def relative_improvement(model_err: float, baseline_err: float) -> float:
    """100 * (baseline - model) / baseline. Positive = model beats baseline."""
    if baseline_err == 0:
        return np.nan
    return 100 * (baseline_err - model_err) / baseline_err
