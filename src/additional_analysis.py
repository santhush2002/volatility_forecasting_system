"""
additional_analysis.py
----------------------
Stage 12 - the additional investigation.

PRIMARY: Do the models behave differently in calm versus turbulent markets
(volatility regimes), and is their ranking stable over time?
SUPPORTING: permutation feature importance (what drives the ML forecasts),
sensitivity to the 1 Dec 2025 extreme observation, and a robustness re-run
on the non-voting share class.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from . import evaluation as ev
from .diagnostics import model_colors
from .exploratory_analysis import save
from .feature_engineering import FEATURES, FEATURE_DESCRIPTIONS


def regime_labels(res) -> pd.Series:
    """Regime of each forecast day, decided at the forecast ORIGIN:
    'High-volatility' if the trailing 22-day mean volatility (known at day t)
    is above its TRAINING-sample median, otherwise 'Low-volatility'.
    The threshold is fixed from training data, so no future data is used."""
    d = res["d"]
    thr = d.loc[:config.TRAIN_END, "lv_m"].median()
    return pd.Series(np.where(d["lv_m"] > thr, "High-volatility", "Low-volatility"),
                     index=d.index), float(np.exp(thr))


def regime_table(res, exclude=None) -> tuple:
    labels, thr = regime_labels(res)
    p = res["pred"][res["pred"]["set"] != "train"].copy()   # all out-of-sample days
    if exclude is not None:
        p = p.drop(index=[pd.Timestamp(x) for x in exclude], errors="ignore")
    p["regime"] = labels.reindex(p.index)
    rows = []
    for reg, sub in p.groupby("regime"):
        for m in res["models"]:
            rows.append({"regime": reg, "days": len(sub), "model": m, "approach": res["approach"][m],
                         "QLIKE": ev.qlike(sub["actual"], sub[m]), "RMSE": ev.rmse(sub["actual"], sub[m]),
                         "mean actual": sub["actual"].mean(), "mean forecast": sub[m].mean()})
    t = pd.DataFrame(rows)
    t["rank in regime (QLIKE)"] = t.groupby("regime")["QLIKE"].rank().astype(int)
    return t, thr


def fig_regimes(res, table, table_excl):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, tab, sub in [(axes[0], table, "all out-of-sample days"),
                         (axes[1], table_excl, "excluding 1 Dec 2025")]:
        piv = tab.pivot(index="model", columns="regime", values="QLIKE").loc[res["models"]]
        n = tab.groupby("regime")["days"].first()
        x = np.arange(len(piv))
        w = 0.38
        for i, reg in enumerate(["Low-volatility", "High-volatility"]):
            ax.bar(x + (i - 0.5) * w, piv[reg], width=w, color=["#90caf9", "#ef9a9a"][i],
                   label=f"{reg} ({n.get(reg, 0)} days)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" on ln-vol", "") for m in piv.index], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("QLIKE loss (lower is better)")
        ax.set_title(sub, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Out-of-sample QLIKE by volatility regime (validation + test periods)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, f"fig12_regime_qlike_{res['share_class']}")


def rolling_loss(res, window=22) -> pd.DataFrame:
    p = res["pred"][res["pred"]["set"] != "train"]
    out = pd.DataFrame(index=p.index)
    for m in res["models"]:
        out[m] = pd.Series(ev.qlike_series(p["actual"].values, p[m].values), index=p.index)
    return out.rolling(window).mean()


def fig_rolling_loss(res, roll):
    cols = model_colors(res)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for m in res["models"]:
        ls = "--" if res["approach"][m] == "Baseline" else "-"
        ax.plot(roll.index, roll[m], color=cols[m], ls=ls, lw=1.4, label=m)
    ax.axvline(pd.Timestamp(config.VAL_END), color="black", lw=0.8)
    ax.text(pd.Timestamp(config.VAL_END), ax.get_ylim()[1] * 0.95, "  test period starts", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Rolling 22-day mean QLIKE (log scale)")
    ax.set_xlabel("Forecast target date")
    ax.set_title("Stability: rolling 22-day QLIKE of each model over the out-of-sample period")
    ax.legend(fontsize=7, ncol=3)
    return save(fig, f"fig13_rolling_qlike_{res['share_class']}")


def event_windows(res) -> pd.DataFrame:
    """Errors around two externally documented shock periods."""
    windows = {"US tariff shock (2-11 Apr 2025)": ("2025-04-02", "2025-04-11"),
               "Post-Cyclone Ditwah (1-5 Dec 2025)": ("2025-12-01", "2025-12-05")}
    rows = []
    for name, (a, b) in windows.items():
        sub = res["pred"].loc[a:b]
        for m in res["models"]:
            rows.append({"event": name, "days": len(sub), "model": m,
                         "mean actual": sub["actual"].mean(), "mean forecast": sub[m].mean(),
                         "QLIKE": ev.qlike(sub["actual"], sub[m]), "RMSE": ev.rmse(sub["actual"], sub[m])})
    return pd.DataFrame(rows)


def outlier_sensitivity(res, date="2025-12-01") -> pd.DataFrame:
    p = res["pred"][res["pred"]["set"] == "test"]
    q = p.drop(index=pd.Timestamp(date))
    rows = []
    for m in res["models"]:
        rows.append({"model": m, "test QLIKE (all)": ev.qlike(p["actual"], p[m]),
                     "test QLIKE (excl. 1 Dec)": ev.qlike(q["actual"], q[m]),
                     "test RMSE (all)": ev.rmse(p["actual"], p[m]),
                     "test RMSE (excl. 1 Dec)": ev.rmse(q["actual"], q[m])})
    t = pd.DataFrame(rows).set_index("model")
    t["rank (all)"] = t["test QLIKE (all)"].rank().astype(int)
    t["rank (excl. 1 Dec)"] = t["test QLIKE (excl. 1 Dec)"].rank().astype(int)
    return t


def permutation_importance(res, n_repeats=30, exclude=("2025-12-01",)) -> pd.DataFrame:
    """Increase in test QLIKE when one feature is randomly shuffled.
    Shuffling breaks the link between that feature and the target; if the
    loss rises a lot, the model relies on the feature. This measures
    PREDICTIVE reliance, not economic causation."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    # The single 1 Dec 2025 outlier dominates test QLIKE, so importance is
    # measured on the remaining test days to reflect typical behaviour.
    test = res["test"].drop(index=[pd.Timestamp(x) for x in exclude], errors="ignore")
    rows = []
    for name, det in res["details"]["ml"].items():
        model, s = det["fits"]["test"]["model"], det["fits"]["test"]["smear"]
        X = test[FEATURES].copy()
        base = ev.qlike(test["y"].values, np.exp(model.predict(X)) * s)
        for f in FEATURES:
            incs = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[f] = rng.permutation(Xp[f].values)
                incs.append(ev.qlike(test["y"].values, np.exp(model.predict(Xp)) * s) - base)
            rows.append({"model": name, "feature": f, "mean QLIKE increase": np.mean(incs),
                         "std": np.std(incs)})
    t = pd.DataFrame(rows)
    t["description"] = t["feature"].map(FEATURE_DESCRIPTIONS)
    return t


def fig_importance(res, imp):
    ml = list(res["details"]["ml"].keys())
    cols = model_colors(res)
    order = (imp.groupby("feature")["mean QLIKE increase"].mean().sort_values().index)
    fig, axes = plt.subplots(1, len(ml), figsize=(13, 5), sharey=True)
    for ax, m in zip(axes, ml):
        s = imp[imp["model"] == m].set_index("feature").loc[order]
        ax.barh(s.index, s["mean QLIKE increase"], xerr=s["std"], color=cols[m], alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(m)
        ax.set_xlabel("Increase in test QLIKE when shuffled")
    fig.suptitle("Permutation feature importance, test period excl. 1 Dec 2025 (30 shuffles)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, f"fig14_permutation_importance_{res['share_class']}")


def ridge_coefficients(res) -> pd.DataFrame:
    pipe = res["details"]["ml"]["Ridge regression"]["fits"]["test"]["model"]
    coef = pipe.named_steps["model"].coef_
    return (pd.DataFrame({"feature": FEATURES, "standardised coefficient": coef,
                          "description": [FEATURE_DESCRIPTIONS[f] for f in FEATURES]})
            .sort_values("standardised coefficient", key=np.abs, ascending=False)
            .reset_index(drop=True))
