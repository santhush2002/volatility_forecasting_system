"""
diagnostics.py
--------------
Stages 13-15 - comparison tables, Diebold-Mariano tests, residual diagnostics
and forecast plots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import config
from . import evaluation as ev
from .exploratory_analysis import save, shade_splits

PALETTE = {"Baseline": ["#9e9e9e", "#616161"], "Traditional": ["#1565c0", "#00897b"],
           "Machine learning": ["#c62828", "#ef6c00", "#6a1b9a"]}


def model_colors(res):
    cols, used = {}, {k: 0 for k in PALETTE}
    for m in res["models"]:
        a = res["approach"][m]
        cols[m] = PALETTE[a][used[a]]
        used[a] += 1
    return cols


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def comparison_table(res) -> pd.DataFrame:
    p = res["pred"]
    rows = []
    for m in res["models"]:
        row = {"model": m, "approach": res["approach"][m]}
        for s, lab in [("train", "train"), ("validation", "val"), ("test", "test")]:
            sub = p[p["set"] == s]
            met = ev.all_metrics(sub["actual"].values, sub[m].values)
            for k in ("QLIKE", "RMSE", "MAE"):
                row[f"{lab} {k}"] = met[k]
            if s == "test":
                row["test MZ R2"] = met["MZ_R2"]
                row["test mean forecast"] = met["mean_forecast"]
        rows.append(row)
    t = pd.DataFrame(rows).set_index("model")
    t["val rank (QLIKE)"] = t["val QLIKE"].rank().astype(int)
    t["test rank (QLIKE)"] = t["test QLIKE"].rank().astype(int)
    return t


def improvement_table(comp: pd.DataFrame, baseline="Historical mean (22-day)") -> pd.DataFrame:
    rows = []
    for m in comp.index:
        rows.append({"model": m,
                     "QLIKE improvement vs 22-day mean (%)": ev.relative_improvement(
                         comp.loc[m, "test QLIKE"], comp.loc[baseline, "test QLIKE"]),
                     "RMSE improvement vs 22-day mean (%)": ev.relative_improvement(
                         comp.loc[m, "test RMSE"], comp.loc[baseline, "test RMSE"]),
                     "QLIKE improvement vs naive (%)": ev.relative_improvement(
                         comp.loc[m, "test QLIKE"], comp.loc["Naive (sigma_t)", "test QLIKE"])})
    return pd.DataFrame(rows).set_index("model")


def dm_table(res, reference: str, subset="test", exclude=None) -> pd.DataFrame:
    """DM test of every model against `reference` on the chosen subset.
    Negative statistic = the model has LOWER loss than the reference."""
    p = res["pred"][res["pred"]["set"] == subset]
    if exclude is not None:
        p = p.drop(index=[pd.Timestamp(x) for x in exclude], errors="ignore")
    y = p["actual"].values
    rows = []
    for m in res["models"]:
        if m == reference:
            continue
        q_stat, q_p = ev.diebold_mariano(ev.qlike_series(y, p[m].values),
                                         ev.qlike_series(y, p[reference].values))
        s_stat, s_p = ev.diebold_mariano((y - p[m].values) ** 2, (y - p[reference].values) ** 2)
        rows.append({"model": m, "DM stat (QLIKE)": q_stat, "p (QLIKE)": q_p,
                     "DM stat (squared error)": s_stat, "p (squared error)": s_p})
    return pd.DataFrame(rows).set_index("model")


def arima_params_table(res) -> pd.DataFrame:
    r = res["details"]["arima"]["fits"]["test"]["res"]
    names = r.model.param_names
    return pd.DataFrame({"estimate": r.params, "std. error": r.bse, "p-value": r.pvalues}, index=names)


def garch_params_table(res) -> pd.DataFrame:
    r = res["details"]["garch"]["runs"][res["details"]["garch"]["name"]]["test"]["res"]
    t = pd.DataFrame({"estimate": r.params, "std. error": r.std_err, "p-value": r.pvalues})
    if "alpha[1]" in t.index and "beta[1]" in t.index:
        g = t.loc["gamma[1]", "estimate"] / 2 if "gamma[1]" in t.index else 0
        t.loc["persistence (alpha+beta+gamma/2)", "estimate"] = (
            t.loc["alpha[1]", "estimate"] + t.loc["beta[1]", "estimate"] + g)
    return t


def residual_tests(res) -> pd.DataFrame:
    """Ljung-Box on traditional-model residuals over the training sample
    (where the parameters were estimated)."""
    rows = []
    ar = res["details"]["arima"]["fits"]["val"]
    d = res["d"]
    tr = d.index[d.index <= pd.Timestamp(config.TRAIN_END)]
    e = (np.log(d.loc[tr, "y"]) - ar["log_mean"].loc[tr]).values
    for lag in (5, 10, 20):
        lb = acorr_ljungbox(e, lags=[lag], return_df=True)
        rows.append({"model": res["details"]["arima"]["name"], "residual": "one-step ln-vol error",
                     "lag": lag, "LB stat": lb["lb_stat"].iloc[0], "p-value": lb["lb_pvalue"].iloc[0]})
    gname = res["details"]["garch"]["name"]
    gr = res["details"]["garch"]["runs"][gname]["val"]["res"]
    z = gr.std_resid.dropna()
    for lab, s in [("standardised", z), ("squared standardised", z ** 2)]:
        for lag in (5, 10):
            lb = acorr_ljungbox(s, lags=[lag], return_df=True)
            rows.append({"model": gname, "residual": lab, "lag": lag,
                         "LB stat": lb["lb_stat"].iloc[0], "p-value": lb["lb_pvalue"].iloc[0]})
    out = pd.DataFrame(rows)
    out["autocorrelation left? (5%)"] = np.where(out["p-value"] < 0.05, "yes", "no")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_arima_residuals(res):
    ar = res["details"]["arima"]["fits"]["val"]
    d = res["d"]
    tr = d.index[d.index <= pd.Timestamp(config.TRAIN_END)]
    e = np.log(d.loc[tr, "y"]) - ar["log_mean"].loc[tr]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(e.index, e, color="#1565c0", lw=0.8)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Residuals over time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("ln actual - ln forecast")
    axes[0].tick_params(axis="x", rotation=30)
    plot_acf(e, lags=20, zero=False, ax=axes[1])
    axes[1].set_title("Residual ACF")
    axes[1].set_xlabel("Lag (days)")
    axes[2].hist(e, bins=30, density=True, color="#90caf9", edgecolor="white")
    x = np.linspace(e.min(), e.max(), 200)
    axes[2].plot(x, stats.norm.pdf(x, e.mean(), e.std()), color="#c62828")
    axes[2].set_title("Residual distribution")
    axes[2].set_xlabel("Residual (log points)")
    fig.suptitle(f"{res['details']['arima']['name']} residual diagnostics (training sample)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, f"fig09_arima_residuals_{res['share_class']}")


def fig_garch_diagnostics(res):
    gname = res["details"]["garch"]["name"]
    gr = res["details"]["garch"]["runs"][gname]["val"]["res"]
    z = gr.std_resid.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(gr.conditional_volatility.index, gr.conditional_volatility, color="#00897b",
                 label="GARCH conditional sigma")
    axes[0].plot(gr.resid.index, gr.resid.abs(), color="#bdbdbd", lw=0.6, label="|return|")
    axes[0].set_title("Conditional volatility vs |return|")
    axes[0].set_ylabel("% per day")
    axes[0].legend(fontsize=7)
    axes[0].tick_params(axis="x", rotation=30)
    plot_acf(z ** 2, lags=20, zero=False, ax=axes[1])
    axes[1].set_title("ACF of squared standardised residuals")
    axes[1].set_xlabel("Lag (days)")
    stats.probplot(z, dist=stats.t(gr.params["nu"]), plot=axes[2])
    axes[2].get_lines()[0].set(markersize=3, color="#00897b")
    axes[2].set_title(f"Q-Q vs Student-t (nu={gr.params['nu']:.1f})")
    fig.suptitle(f"{gname} diagnostics (training sample)", fontweight="bold")
    fig.tight_layout()
    return save(fig, f"fig10_garch_diagnostics_{res['share_class']}")


def fig_forecasts_test(res, which: str, fignum: str):
    """Actual vs forecasts on the test period for one family + baselines."""
    p = res["pred"][res["pred"]["set"] == "test"]
    cols = model_colors(res)
    fam = [m for m in res["models"] if res["approach"][m] == which]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(p.index, p["actual"], color="black", lw=1.1, marker="o", ms=2.5, label="Actual")
    ax.plot(p.index, p["Historical mean (22-day)"], color="#9e9e9e", ls="--", lw=1.4,
            label="Baseline: 22-day historical mean")
    for m in fam:
        ax.plot(p.index, p[m], color=cols[m], lw=1.5, label=m)
    ax.set_ylim(0, min(4.5, p["actual"].max() * 1.05))
    ax.annotate(f"1 Dec 2025 actual = {p.loc['2025-12-01', 'actual']:.1f}% (off scale)",
                (pd.Timestamp("2025-12-01"), 4.4), xytext=(-200, -8), textcoords="offset points",
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_title(f"Test period (Aug-Dec 2025): one-day-ahead forecasts, {which.lower()} models")
    ax.set_ylabel("Daily Parkinson volatility (%)")
    ax.set_xlabel("Forecast target date")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    return save(fig, f"fig{fignum.lower()}_test_forecasts_{which.split()[0].lower()}_{res['share_class']}")


def fig_full_forecast(res, best: str):
    """Whole history with split boundaries, best model and ARIMA interval."""
    p = res["pred"]
    g = res["g"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    shade_splits(ax)
    ax.plot(g.index, g["park_vol"], color="black", lw=0.7, label="Actual volatility")
    oos = p[p["set"] != "train"]
    ax.plot(oos.index, oos[best], color="#c62828", lw=1.4, label=f"{best} (out-of-sample)")
    ax.plot(oos.index, oos["Historical mean (22-day)"], color="#757575", ls="--", lw=1.1,
            label="Baseline: 22-day mean")
    t = p[p["set"] == "test"]
    ax.fill_between(t.index, t["ARIMA_PI80_low"], t["ARIMA_PI80_high"], color="#1565c0", alpha=0.18,
                    label=f"80% prediction interval ({res['details']['arima']['name']})")
    ax.set_ylim(0, 6.5)
    ax.set_ylabel("Daily Parkinson volatility (%)")
    ax.set_xlabel("Date")
    ax.set_title("Full sample with training / validation / test periods and out-of-sample forecasts")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    return save(fig, f"fig08_full_forecast_{res['share_class']}")


def fig_ml_diagnostics(res):
    p = res["pred"][res["pred"]["set"] == "test"]
    ml = [m for m in res["models"] if res["approach"][m] == "Machine learning"]
    cols = model_colors(res)
    fig, axes = plt.subplots(2, len(ml), figsize=(13, 7))
    for i, m in enumerate(ml):
        ax = axes[0, i]
        ax.scatter(p[m], p["actual"], s=12, color=cols[m], alpha=0.7)
        lim = [0, 4]
        ax.plot(lim, lim, color="black", lw=0.8, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{m}: actual vs forecast")
        ax.set_xlabel("Forecast (%)")
        ax.set_ylabel("Actual (%)")
        ax = axes[1, i]
        e = p["actual"] - p[m]
        ax.bar(p.index, e, color=np.where(e > 0, "#c62828", "#1565c0"), width=1.5)
        ax.set_ylim(-2, 3)
        ax.set_title("Error over time (actual - forecast)")
        ax.set_ylabel("Percentage points")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Machine-learning test-period diagnostics (1 Dec 2025 outlier is off scale)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, f"fig11_ml_diagnostics_{res['share_class']}")
