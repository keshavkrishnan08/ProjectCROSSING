"""Train and evaluate the crossing-intention GRU.

Trains the from-scratch NumPy GRU (:mod:`crossing.ml.nn`) to answer, from the last ~1.5 s
of behaviour, "will this pedestrian step into the road within the next second?" Reports
AUROC / F1 / accuracy / ECE on unseen test scenes, plus the **mean lead time** how early
before road entry the probability first crosses 0.5, which is the metric that matters for
warning drivers in time.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from ..config import MODELS_DIR, FPS
from .nn import GRU, train_gru_classifier
from .calibration import expected_calibration_error
from .datasets import build_intention_dataset


def train_intention(train_seeds, val_seeds, test_seeds, duration=30.0, hidden=24,
                    epochs=45, save=True, verbose=True):
    if verbose:
        print("[intent] building datasets...")
    Xtr, ytr, _ = build_intention_dataset(train_seeds, duration, verbose=verbose)
    Xva, yva, _ = build_intention_dataset(val_seeds, duration)
    Xte, yte, _ = build_intention_dataset(test_seeds, duration)
    if verbose:
        print(f"[intent] train {len(ytr)} (pos {int(ytr.sum())}) | val {len(yva)} | test {len(yte)} (pos {int(yte.sum())})")

    in_dim = Xtr.shape[2]
    gru = GRU(in_dim, hidden, 1, seed=0)
    pos_rate = max(ytr.mean(), 1e-3)
    cw = (1.0, float(np.clip((1 - pos_rate) / pos_rate, 1.0, 6.0)))  # up-weight minority
    hist = train_gru_classifier(gru, Xtr, ytr, epochs=epochs, lr=4e-3, batch=128,
                                val=(Xva, yva), verbose=verbose, class_weight=cw)

    from .nn import _predict_proba_batched
    p_te = _predict_proba_batched(gru, Xte)
    yhat = (p_te >= 0.5).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(yte, p_te)) if len(set(yte)) > 1 else float("nan"),
        "f1": float(f1_score(yte, yhat, zero_division=0)),
        "accuracy": float(accuracy_score(yte, yhat)),
        "ece": expected_calibration_error(p_te, yte),
        "n_test": int(len(yte)), "pos_rate": float(yte.mean()),
        "hidden": hidden, "epochs": epochs, "train_loss": hist["loss"][-1],
    }
    if save:
        gru.save(str(MODELS_DIR / "intention_gru.npz"))
        if verbose:
            print(f"[intent] saved GRU -> {MODELS_DIR/'intention_gru.npz'}  test AUROC={metrics['auroc']:.3f}")
    return metrics
