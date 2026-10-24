"""
pipeline.py
-----------
Runs the complete modelling workflow for one share class and returns every
object needed for tables, figures and the report.

Order of operations (the order matters to avoid using test information):
 1. clean data and build features
 2. chronological split
 3. choose ARIMA order by AIC on TRAINING data
 4. choose GARCH variant by VALIDATION QLIKE
 5. tune ML hyper-parameters by VALIDATION QLIKE
 6. re-estimate every chosen model on TRAINING + VALIDATION
 7. forecast the TEST period once, with fixed parameters
"""
import warnings

import numpy as np
import pandas as pd

from . import config
from . import data_processing as dp
from . import feature_engineering as fe
from . import modelling as md
from .evaluation import qlike



def run_models(share_class: str = config.MAIN_CLASS, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    g = dp.build_dataset(share_class)
    d = fe.build_features(g)
    train, val, test = fe.split(d)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    stage_preds = {"val": pd.DataFrame(index=d.index), "test": pd.DataFrame(index=d.index)}
    approach, details = {}, {}

    # ---- baselines ----------------------------------------------------------
    for name, col in md.BASELINES.items():
        for st in stage_preds:
            stage_preds[st][name] = d[col]
        approach[name] = "Baseline"

    # ---- ARIMA ----------------------------------------------------------------
    arima_sel = md.select_arima(g["log_vol"], train.index)
    order = tuple(arima_sel.loc[0, "order"])
    arima_name = f"ARIMA{order} on ln-vol"
    ar = md.run_arima(g, d, train, val, test, order)
    for st in stage_preds:
        stage_preds[st][arima_name] = ar[st]["pred"].reindex(d.index)
    approach[arima_name] = "Traditional"
    details["arima"] = {"selection": arima_sel, "order": order, "fits": ar, "name": arima_name}
    log(f"[{share_class}] ARIMA order by AIC: {order}")

    # ---- GARCH family: pick variant on validation -----------------------------
    garch_runs, garch_val = {}, []
    for spec in md.GARCH_SPECS:
        out = md.run_garch(g, d, spec)
        garch_runs[spec] = out
        p = out["val"]["pred"].loc[val.index]
        res = out["val"]["res"]
        garch_val.append({"specification": spec, "train log-lik": res.loglikelihood,
                          "train AIC": res.aic, "train BIC": res.bic,
                          "val QLIKE": qlike(val["y"].values, p.values)})
    garch_val = pd.DataFrame(garch_val).sort_values("val QLIKE").reset_index(drop=True)
    garch_name = garch_val.loc[0, "specification"]
    for st in stage_preds:
        stage_preds[st][garch_name] = garch_runs[garch_name][st]["pred"].reindex(d.index)
    approach[garch_name] = "Traditional"
    details["garch"] = {"selection": garch_val, "name": garch_name, "runs": garch_runs}
    log(f"[{share_class}] GARCH variant by validation QLIKE: {garch_name}")

    # ---- machine learning -----------------------------------------------------
    details["ml"] = {}
    for name in md.ML_GRIDS:
        tab, best = md.tune_ml(name, train, val)
        out = md.run_ml(name, best, d, train, val, test)
        for st in stage_preds:
            stage_preds[st][name] = out[st]["pred"]
        approach[name] = "Machine learning"
        details["ml"][name] = {"tuning": tab, "best": best, "fits": out}
        log(f"[{share_class}] {name} best params: {best}")

    # ---- combine: train/val rows from 'val' stage, test rows from 'test' stage --
    models = list(approach.keys())
    pred = stage_preds["val"].copy()
    pred.loc[test.index, models] = stage_preds["test"].loc[test.index, models]
    pred.insert(0, "actual", d["y"])
    pred.insert(0, "set", np.select([pred.index <= pd.Timestamp(config.TRAIN_END),
                                     pred.index <= pd.Timestamp(config.VAL_END)],
                                    ["train", "validation"], "test"))
    # ARIMA 80% prediction interval for the test period (fitted on train+val)
    pred["ARIMA_PI80_low"] = ar["test"]["lo"].reindex(d.index)
    pred["ARIMA_PI80_high"] = ar["test"]["hi"].reindex(d.index)

    return {"share_class": share_class, "g": g, "d": d, "train": train, "val": val, "test": test,
            "pred": pred, "models": models, "approach": approach, "details": details}
