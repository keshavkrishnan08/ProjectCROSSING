"""Progressively stronger risk baselines, for an honest benchmark.

A learned model only earns its keep if it beats sensible baselines. CROSSING benchmarks
five, each producing a risk score in [0,1] from an interaction's features:

1. **distance**      closer = riskier (the crudest surrogate).
2. **ttc**           inverse time-to-collision threshold.
3. **ttc+motion**    TTC combined with closing speed.
4. **physics**       the interpretable logistic fusion (:class:`PhysicsRisk`).
5. **learned**       the calibrated gradient-boosted model.

The evaluation reports precision / recall / F1 / AUROC / AUPRC for every one of these.
"""
from __future__ import annotations

import numpy as np

from ..risk.engine import PhysicsRisk


def distance_score(f: dict) -> float:
    d = f.get("distance", 99.0)
    return float(np.clip((12.0 - d) / 12.0, 0, 1))


def ttc_score(f: dict) -> float:
    inv = f.get("inv_ttc", 0.0)                 # 1/ttc, 0 if diverging/miss
    return float(np.clip(inv / (1.0 / 0.8), 0, 1))   # ttc<=0.8s -> 1


def ttc_motion_score(f: dict) -> float:
    inv = f.get("inv_ttc", 0.0)
    close = f.get("closing_speed", 0.0)
    return float(np.clip(0.6 * inv / (1.0 / 0.8) + 0.4 * close / 10.0, 0, 1))


def physics_score(f: dict) -> float:
    return PhysicsRisk().score(f)[0] / 100.0


BASELINES = {
    "distance": distance_score,
    "ttc": ttc_score,
    "ttc+motion": ttc_motion_score,
    "physics": physics_score,
}


def score_all(feat_dicts, fn) -> np.ndarray:
    return np.array([fn(f) for f in feat_dicts], float)
