"""
main.py
-------
Runs the whole project from raw data to every table, figure and prediction
file. Usage (from the project folder):

    python main.py

Everything is written to outputs/. The notebook in notebooks/ calls the same
functions step by step with explanations, so both give identical numbers.
"""
import json
import warnings

import numpy as np
import pandas as pd

from src import config
from src import data_processing as dp
from src import exploratory_analysis as eda
from src import feature_engineering as fe
from src import diagnostics as dg
from src import additional_analysis as aa
from src.pipeline import run_models

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)


def save_table(df, name, index=True):
    df.to_csv(config.TAB_DIR / f"{name}.csv", index=index, float_format="%.6g")
    return df


def main():
    tables = {}

    # ------------------------------------------------------------------ data
    raw = dp.load_raw()
    tables["T01_raw_overview"] = save_table(dp.raw_overview(raw), "T01_raw_overview", index=False)
    tables["T02_quality_N"] = save_table(dp.quality_report(raw, "N"), "T02_quality_N", index=False)
    tables["T02b_quality_X"] = save_table(dp.quality_report(raw, "X"), "T02b_quality_X", index=False)

    # ------------------------------------------------------- main share class
    res = run_models(config.MAIN_CLASS)
    g, d, train, val, test = res["g"], res["d"], res["train"], res["val"], res["test"]
    g_x = dp.build_dataset(config.ROBUST_CLASS)

    tables["T03_corporate_actions"] = save_table(dp.corporate_action_table(g, "N"),
                                                 "T03_corporate_actions", index=False)
    tables["T04_split"] = save_table(fe.split_table(train, val, test), "T04_split", index=False)
    tables["T05_descriptive"] = save_table(eda.descriptive_stats(g), "T05_descriptive")
    tables["T06_period_stats"] = save_table(eda.period_stats(g), "T06_period_stats")
    tables["T07_stationarity"] = save_table(eda.stationarity_tests(g), "T07_stationarity")
    tables["T08_dependence"] = save_table(eda.dependence_tests(g), "T08_dependence", index=False)
    kw_h, kw_p = eda.weekday_test(g)
    feat_desc = pd.DataFrame({"feature": fe.FEATURES,
                              "description": [fe.FEATURE_DESCRIPTIONS[f] for f in fe.FEATURES]})
    tables["T09_features"] = save_table(feat_desc, "T09_features", index=False)

    # EDA figures
    eda.fig_price_history(g, g_x)
    eda.fig_returns(g)
    eda.fig_return_distribution(g)
    eda.fig_target(g)
    eda.fig_target_distribution(g)
    eda.fig_acf(g)
    eda.fig_correlation(train)

    # model selection tables
    tables["T10_arima_selection"] = save_table(res["details"]["arima"]["selection"].head(10),
                                               "T10_arima_selection", index=False)
    tables["T11_arima_params"] = save_table(dg.arima_params_table(res), "T11_arima_params")
    tables["T12_garch_selection"] = save_table(res["details"]["garch"]["selection"],
                                               "T12_garch_selection", index=False)
    tables["T13_garch_params"] = save_table(dg.garch_params_table(res), "T13_garch_params")
    for name, det in res["details"]["ml"].items():
        key = "T14_tuning_" + name.split()[0].lower()
        tables[key] = save_table(det["tuning"], key, index=False)

    # comparison
    comp = dg.comparison_table(res)
    tables["T15_model_comparison"] = save_table(comp, "T15_model_comparison")
    tables["T16_improvement"] = save_table(dg.improvement_table(comp), "T16_improvement")
    best_val = comp["val QLIKE"].idxmin()
    best_test = comp["test QLIKE"].idxmin()
    tables["T17_dm_vs_hist"] = save_table(dg.dm_table(res, "Historical mean (22-day)"), "T17_dm_vs_hist")
    tables["T18_dm_vs_bestval"] = save_table(dg.dm_table(res, best_val), "T18_dm_vs_bestval")
    tables["T19_residual_tests"] = save_table(dg.residual_tests(res), "T19_residual_tests", index=False)

    # model figures
    dg.fig_full_forecast(res, best_val)
    dg.fig_arima_residuals(res)
    dg.fig_garch_diagnostics(res)
    dg.fig_forecasts_test(res, "Traditional", "15a")
    dg.fig_forecasts_test(res, "Machine learning", "15b")
    dg.fig_ml_diagnostics(res)

    # predictions file (dates aligned)
    res["pred"].to_csv(config.PRED_DIR / "predictions_all_sets_N.csv", float_format="%.6f")
    res["pred"][res["pred"]["set"] == "test"].to_csv(config.PRED_DIR / "test_actual_vs_predicted_N.csv",
                                                     float_format="%.6f")

    # --------------------------------------------------- additional analysis
    reg, thr = aa.regime_table(res)
    tables["T20_regimes"] = save_table(reg, "T20_regimes", index=False)
    reg_x, _ = aa.regime_table(res, exclude=["2025-12-01"])
    tables["T20b_regimes_excl_outlier"] = save_table(reg_x, "T20b_regimes_excl_outlier", index=False)
    aa.fig_regimes(res, reg, reg_x)
    roll = aa.rolling_loss(res)
    aa.fig_rolling_loss(res, roll)
    tables["T21_event_windows"] = save_table(aa.event_windows(res), "T21_event_windows", index=False)
    tables["T22_outlier_sensitivity"] = save_table(aa.outlier_sensitivity(res), "T22_outlier_sensitivity")
    tables["T22b_dm_vs_hist_excl_outlier"] = save_table(
        dg.dm_table(res, "Historical mean (22-day)", exclude=["2025-12-01"]), "T22b_dm_vs_hist_excl_outlier")
    imp = aa.permutation_importance(res)
    tables["T23_permutation_importance"] = save_table(imp, "T23_permutation_importance", index=False)
    aa.fig_importance(res, imp)
    tables["T24_ridge_coefficients"] = save_table(aa.ridge_coefficients(res), "T24_ridge_coefficients",
                                                  index=False)

    # ------------------------------------------- robustness: non-voting share
    res_x = run_models(config.ROBUST_CLASS)
    comp_x = dg.comparison_table(res_x)
    tables["T25_robustness_X"] = save_table(comp_x, "T25_robustness_X")
    res_x["pred"].to_csv(config.PRED_DIR / "predictions_all_sets_X.csv", float_format="%.6f")

    # all tables in one Excel workbook for convenience
    with pd.ExcelWriter(config.TAB_DIR / "all_tables.xlsx") as xw:
        for k, v in tables.items():
            v.to_excel(xw, sheet_name=k[:31])

    # key numbers for the report (single source of truth)
    summary = {
        "arima_order": list(res["details"]["arima"]["order"]),
        "arima_name": res["details"]["arima"]["name"],
        "garch_name": res["details"]["garch"]["name"],
        "ml_best_params": {k: {kk: (None if vv is None else (float(vv) if isinstance(vv, (float, np.floating)) else vv))
                               for kk, vv in v["best"].items()} for k, v in res["details"]["ml"].items()},
        "best_val_model": best_val, "best_test_model": best_test,
        "weekday_kruskal": {"H": kw_h, "p": kw_p},
        "regime_threshold_vol_pct": thr,
        "garch_calibration_test": {k: res["details"]["garch"]["runs"][res["details"]["garch"]["name"]]["test"][k]
                                   for k in ("calib_a", "calib_b", "smear")},
        "X_best_val": comp_x["val QLIKE"].idxmin(), "X_best_test": comp_x["test QLIKE"].idxmin(),
        "X_arima": res_x["details"]["arima"]["name"], "X_garch": res_x["details"]["garch"]["name"],
    }
    with open(config.MODEL_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Model comparison (COMB.N0000) ===")
    print(comp.round(4).to_string())
    print("\nBest on validation:", best_val, "| best on test:", best_test)
    print(json.dumps(summary, indent=2, default=str))
    return res, res_x, tables, summary


if __name__ == "__main__":
    main()
