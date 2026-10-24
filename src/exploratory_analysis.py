"""
exploratory_analysis.py
-----------------------
Stages 5-6 - exploratory plots, descriptive statistics and statistical tests.

Each figure answers one question (written in its docstring). Figures are
saved as 200-dpi PNG files in outputs/figures/.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from . import config
from .feature_engineering import FEATURES

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold",
                     "axes.titlesize": 12})

SPLIT_COLORS = {"Training": "#e8eef7", "Validation": "#fdf1dc", "Test": "#e6f4ea"}


def save(fig, name):
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def shade_splits(ax, start=None):
    """Background colours for the training / validation / test periods."""
    tr_end, va_end = pd.Timestamp(config.TRAIN_END), pd.Timestamp(config.VAL_END)
    start = start or pd.Timestamp("2024-01-01")
    ax.axvspan(start, tr_end, color=SPLIT_COLORS["Training"], zorder=0, label="Training")
    ax.axvspan(tr_end, va_end, color=SPLIT_COLORS["Validation"], zorder=0, label="Validation")
    ax.axvspan(va_end, pd.Timestamp("2025-12-31"), color=SPLIT_COLORS["Test"], zorder=0, label="Test")


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_stats(g: pd.DataFrame) -> pd.DataFrame:
    cols = {"close": "Close price (Rs.)", "ret": "Adjusted daily log return (%)",
            "park_vol": "Parkinson daily volatility (%)", "volume": "Share volume (No.)"}
    rows = []
    for c, label in cols.items():
        s = g[c].dropna()
        rows.append({"variable": label, "n": len(s), "mean": s.mean(), "std": s.std(),
                     "min": s.min(), "median": s.median(), "max": s.max(),
                     "skewness": stats.skew(s), "excess kurtosis": stats.kurtosis(s)})
    return pd.DataFrame(rows).set_index("variable")


def period_stats(g: pd.DataFrame) -> pd.DataFrame:
    """Return / volatility statistics by year and by split period."""
    periods = {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
               "Training period": ("2024-01-01", config.TRAIN_END),
               "Validation period": ("2025-03-01", config.VAL_END),
               "Test period": ("2025-08-01", "2025-12-31")}
    rows = []
    for name, (a, b) in periods.items():
        s = g.loc[a:b]
        rows.append({"period": name, "days": len(s),
                     "price change (%)": 100 * (s["close"].iloc[-1] / s["close"].iloc[0] - 1),
                     "mean return (%/day)": s["ret"].mean(), "std return (%/day)": s["ret"].std(),
                     "mean Parkinson vol (%)": s["park_vol"].mean(),
                     "median Parkinson vol (%)": s["park_vol"].median(),
                     "annualised vol (%)": s["ret"].std() * np.sqrt(242)})
    return pd.DataFrame(rows).set_index("period")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def stationarity_tests(g: pd.DataFrame, end=config.TRAIN_END) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) on the TRAINING sample.
    Decisions about differencing must not be based on test-period data."""
    series = {"log close price": np.log(g["close"]),
              "adjusted log return": g["ret"],
              "log Parkinson volatility": g["log_vol"]}
    rows = []
    for name, s in series.items():
        s = s.loc[:end].dropna()
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # KPSS p-value is truncated at 0.01/0.10
            k_stat, k_p, *_ = kpss(s, regression="c", nlags="auto")
        if adf_p < 0.05 and k_p > 0.05:
            verdict = "stationary (both tests agree)"
        elif adf_p >= 0.05 and k_p <= 0.05:
            verdict = "non-stationary (both tests agree)"
        else:
            verdict = "tests disagree - inspect plots"
        rows.append({"series": name, "ADF stat": adf_stat, "ADF p": adf_p,
                     "KPSS stat": k_stat, "KPSS p": k_p, "conclusion (5%)": verdict})
    return pd.DataFrame(rows).set_index("series")


def dependence_tests(g: pd.DataFrame, end=config.TRAIN_END) -> pd.DataFrame:
    """Is there exploitable structure? Ljung-Box (H0: no autocorrelation up to
    lag m), ARCH-LM (H0: no ARCH effects), Jarque-Bera (H0: normality)."""
    r = g["ret"].loc[:end].dropna()
    lv = g["log_vol"].loc[:end].dropna()
    rows = []
    for name, s in [("adjusted return", r), ("squared return", r ** 2), ("log Parkinson vol", lv)]:
        lb = acorr_ljungbox(s, lags=[5, 10], return_df=True)
        for lag in (5, 10):
            rows.append({"test": f"Ljung-Box Q({lag})", "series": name,
                         "statistic": lb.loc[lag, "lb_stat"], "p-value": lb.loc[lag, "lb_pvalue"]})
    lm, lm_p, _, _ = het_arch(r - r.mean(), nlags=5)
    rows.append({"test": "ARCH-LM (5 lags)", "series": "adjusted return", "statistic": lm, "p-value": lm_p})
    jb = stats.jarque_bera(r)
    rows.append({"test": "Jarque-Bera", "series": "adjusted return",
                 "statistic": jb.statistic, "p-value": jb.pvalue})
    out = pd.DataFrame(rows)
    out["reject H0 at 5%?"] = np.where(out["p-value"] < 0.05, "yes", "no")
    return out


def weekday_test(g: pd.DataFrame, end=config.TRAIN_END):
    """Kruskal-Wallis test: does volatility differ by weekday? (seasonality check)"""
    s = g.loc[:end]
    groups = [grp["park_vol"].values for _, grp in s.groupby(s.index.dayofweek)]
    h, p = stats.kruskal(*groups)
    return float(h), float(p)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_price_history(g_n, g_x):
    """Q: How did the share prices evolve and where are the corporate actions?"""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade_splits(ax)
    ax.plot(g_n.index, g_n["close"], color="#1f4e79", lw=1.6, label="Voting share COMB.N0000")
    ax.plot(g_x.index, g_x["close"], color="#8c8c8c", lw=1.2, label="Non-voting share COMB.X0000")
    for ca in config.CORPORATE_ACTIONS["N"]:
        d = pd.Timestamp(ca["date"])
        ax.axvline(d, color="#b03a2e", ls="--", lw=0.9)
        ax.text(d, ax.get_ylim()[1] * 0.97, " " + ("XD" if ca["type"] == "dividend" else "XR"),
                color="#b03a2e", fontsize=8, va="top")
    ax.set_title("COMB daily closing prices, Jan 2024 - Dec 2025 (unadjusted)")
    ax.set_ylabel("Closing price (Rs.)")
    ax.set_xlabel("Trading date")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    return save(fig, "fig01_price_history")


def fig_returns(g):
    """Q: Are returns calm and constant, or do volatile periods cluster?"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    ax = axes[0]
    ax.bar(g.index, g["ret"], width=1.5, color=np.where(g["ret"] >= 0, "#2e7d32", "#c62828"))
    ax.set_title("Adjusted daily log returns of COMB.N0000 and their rolling 22-day statistics")
    ax.set_ylabel("Daily log return (%)")
    for d, lab in [("2024-05-29", "Rights issue\nannounced"), ("2025-04-07", "US tariff\nsell-off")]:
        ax.annotate(lab, (pd.Timestamp(d), g.loc[d, "ret"]), xytext=(15, -5),
                    textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax = axes[1]
    roll_sd = g["ret"].rolling(config.MONTH).std()
    roll_mu = g["ret"].rolling(config.MONTH).mean()
    ax.plot(g.index, roll_sd, color="#1f4e79", label="Rolling 22-day std. dev. of returns")
    ax.plot(g.index, roll_mu, color="#e67e22", label="Rolling 22-day mean return")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Percent per day")
    ax.set_xlabel("Trading date")
    ax.legend(fontsize=8)
    return save(fig, "fig02_returns_rolling")


def fig_return_distribution(g):
    """Q: Are returns normally distributed (an assumption of simple models)?"""
    r = g["ret"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(r, bins=50, density=True, color="#90a4ae", edgecolor="white", label="Observed")
    x = np.linspace(r.min(), r.max(), 300)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color="#c62828", label="Normal with same mean/std")
    ax.set_title("Distribution of daily returns")
    ax.set_xlabel("Daily log return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax = axes[1]
    stats.probplot(r, dist="norm", plot=ax)
    ax.get_lines()[0].set(markersize=3, color="#1f4e79")
    ax.set_title("Normal Q-Q plot of daily returns")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed return quantiles (%)")
    return save(fig, "fig03_return_distribution")


def fig_target(g):
    """Q: What does the forecasting target look like over time?"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for ax in axes:
        shade_splits(ax)
    axes[0].plot(g.index, g["park_vol"], color="#1f4e79", lw=0.9, label="Daily Parkinson volatility")
    axes[0].plot(g.index, g["park_vol"].rolling(config.MONTH).mean(), color="#e67e22", lw=1.8,
                 label="22-day rolling mean")
    axes[0].set_ylabel("Volatility (% per day)")
    axes[0].set_title("Forecasting target: daily range-based (Parkinson) volatility of COMB.N0000")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].annotate("1 Dec 2025: intraday low Rs 170\n(close Rs 200.25)",
                     (pd.Timestamp("2025-12-01"), g.loc["2025-12-01", "park_vol"]),
                     xytext=(-170, -10), textcoords="offset points", fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.6))
    axes[1].plot(g.index, g["log_vol"], color="#455a64", lw=0.9)
    axes[1].set_ylabel("ln(volatility)")
    axes[1].set_xlabel("Trading date")
    axes[1].set_title("Log volatility (scale used by ARIMA and ML models)", fontsize=10)
    return save(fig, "fig04_target_series")


def fig_target_distribution(g):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].hist(g["park_vol"], bins=45, color="#90a4ae", edgecolor="white")
    axes[0].set_title("Volatility (% per day)")
    axes[0].set_xlabel("Parkinson volatility (%)")
    axes[0].set_ylabel("Number of days")
    axes[1].hist(g["log_vol"], bins=35, color="#1f4e79", edgecolor="white")
    axes[1].set_title("ln(volatility)")
    axes[1].set_xlabel("ln Parkinson volatility")
    s = g.loc[:config.TRAIN_END]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    wd = pd.Categorical(s.index.day_name().str[:3], categories=order, ordered=True)
    sns.boxplot(x=wd, y=s["park_vol"], ax=axes[2], color="#cfd8dc", fliersize=2)
    axes[2].set_title("Volatility by weekday (training)")
    axes[2].set_xlabel("Weekday")
    axes[2].set_ylabel("Parkinson volatility (%)")
    axes[2].set_ylim(0, 4)
    fig.tight_layout()
    return save(fig, "fig05_target_distribution")


def fig_acf(g, end=config.TRAIN_END):
    """Q: Is there memory (autocorrelation) that a model can exploit?"""
    s = g.loc[:end]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    plot_acf(s["ret"].dropna(), lags=25, ax=axes[0, 0], zero=False)
    axes[0, 0].set_title("ACF of daily returns")
    plot_acf((s["ret"].dropna()) ** 2, lags=25, ax=axes[0, 1], zero=False)
    axes[0, 1].set_title("ACF of squared returns")
    plot_acf(s["log_vol"], lags=25, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title("ACF of ln(Parkinson volatility)")
    plot_pacf(s["log_vol"], lags=25, ax=axes[1, 1], zero=False, method="ywm")
    axes[1, 1].set_title("PACF of ln(Parkinson volatility)")
    for ax in axes.flat:
        ax.set_xlabel("Lag (trading days)")
        ax.set_ylabel("Correlation")
    fig.suptitle("Autocorrelation structure, training sample (Feb 2024 - Feb 2025)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "fig06_acf_pacf")


def fig_correlation(train: pd.DataFrame):
    """Q: Which candidate predictors move with next-day volatility?"""
    cols = FEATURES + ["log_y"]
    corr = train[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, cbar_kws={"label": "Pearson correlation", "shrink": 0.75}, ax=ax)
    ax.set_title("Correlation of features (known at day t) with next-day ln volatility\n"
                 "(training sample only)")
    return save(fig, "fig07_feature_correlation")
