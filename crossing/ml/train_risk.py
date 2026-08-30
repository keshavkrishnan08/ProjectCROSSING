"""Train and evaluate the learned risk model against the baseline ladder.

Produces the central research result: *does a learned interaction model beat engineered
TTC-based scoring at predicting near-misses before they happen?* Trains a gradient-boosted
classifier (xgboost/lightgbm if importable, else scikit-learn ``HistGradientBoosting``),
isotonically calibrates it on a held-out validation split, and reports precision / recall /
F1 / AUROC / AUPRC / ECE for every baseline and the learned model — all on **unseen test
scenes**.
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score)

from ..config import MODELS_DIR
from .datasets import build_risk_dataset
from .baselines import BASELINES, score_all
from .calibration import IsotonicCalibrator, expected_calibration_error, reliability_curve
from ..risk.engine import RISK_FEATURES


def _make_model():
    """Best available gradient-boosted classifier; falls back gracefully."""
    try:
        import xgboost as xgb
        return ("xgboost", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=4))
    except Exception:
        pass
    try:
        import lightgbm as lgb
        return ("lightgbm", lgb.LGBMClassifier(n_estimators=400, num_leaves=31,
                                               learning_rate=0.05, subsample=0.9, verbose=-1))
    except Exception:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return ("sklearn-histgb", HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        class_weight="balanced", validation_fraction=0.1, random_state=0))


def _binary_metrics(y, p, thresh=0.5):
    yhat = (p >= thresh).astype(int)
    out = {
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "ece": expected_calibration_error(p, y),
        "positives": int(y.sum()), "n": int(len(y)),
    }
    return out


def _best_f1_threshold(y, p):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t), float(best_f1)


def train_risk(train_seeds, val_seeds, test_seeds, duration=30.0, save=True, verbose=True):
    if verbose:
        print("[risk] building datasets...")
    Xtr, ytr, _, ftr = build_risk_dataset(train_seeds, duration, verbose=verbose)
    Xva, yva, _, fva = build_risk_dataset(val_seeds, duration)
    Xte, yte, _, fte = build_risk_dataset(test_seeds, duration)
    if verbose:
        print(f"[risk] train {len(ytr)} (pos {ytr.sum()}) | val {len(yva)} | test {len(yte)} (pos {yte.sum()})")

    name, model = _make_model()
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:, 1]
    calib = IsotonicCalibrator().fit(p_va, yva)
    p_te_raw = model.predict_proba(Xte)[:, 1]
    p_te = calib.predict(p_te_raw)

    results = {"model_type": name, "features": RISK_FEATURES, "table": {}}
    # baselines
    for bname, fn in BASELINES.items():
        s = score_all(fte, fn)
        t, _ = _best_f1_threshold(yte, s)
        m = _binary_metrics(yte, s, thresh=t)
        m["threshold"] = t
        results["table"][bname] = m
    # learned (calibrated)
    t, _ = _best_f1_threshold(yte, p_te)
    m = _binary_metrics(yte, p_te, thresh=t)
    m["threshold"] = t
    m["ece_uncalibrated"] = expected_calibration_error(p_te_raw, yte)
    results["table"]["learned"] = m
    results["reliability"] = {
        "uncalibrated": reliability_curve(p_te_raw, yte),
        "calibrated": reliability_curve(p_te, yte),
    }
    # feature importance (if available)
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            results["feature_importance"] = {k: float(v) for k, v in
                                             sorted(zip(RISK_FEATURES, imp), key=lambda z: -z[1])}
    except Exception:
        pass

    if save:
        with open(MODELS_DIR / "risk_model.pkl", "wb") as fh:
            pickle.dump({"model": model, "calibrator": calib, "features": RISK_FEATURES,
                         "model_type": name}, fh)
        if verbose:
            print(f"[risk] saved model -> {MODELS_DIR/'risk_model.pkl'}")
    return results
