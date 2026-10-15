"""Risk scoring, from conflict geometry to a calibrated 0-100 danger score.

Two models, one interface:

* **PhysicsRisk** a transparent logistic fusion of danger-aligned factors (low TTC,
  small miss distance, path overlap, closing speed, crossing intention, vehicle speed,
  permissive turn, inattention, braking relief). Because the score is a weighted sum, the
  same computation yields the **explanation** for free: each factor's share of the score
  (the "Low TTC +38 / Trajectory intersection +27 / ..." decomposition).
* **LearnedRisk** a gradient-boosted model (scikit-learn ``HistGradientBoosting``, or
  xgboost/lightgbm if installed) trained on the operational near-miss label and probability-
  calibrated. Used when a model file is present; otherwise the physics model drives risk.

We never pretend the score is ground truth: the learned model is calibrated (see
``crossing.ml.calibration``) so that "risk 80" really does mean ~80% of such situations
become near-misses under the operational definition.
"""
from __future__ import annotations

import math
import pickle

import numpy as np

from ..config import MODELS_DIR, RISK
from ..schema import Interaction, RiskBand

# ordered feature list the learned model consumes
RISK_FEATURES = [
    "inv_ttc", "dcpa", "tcpa", "closing_speed", "path_conflict", "distance",
    "veh_speed", "veh_braking", "veh_turning", "vru_intention", "vru_speed",
    "vru_on_road", "vru_not_looking", "approach_angle", "inv_pet",
]

# interpretable factor weights (logit space). Tuned so the score is well-spread and the
# operational near-miss threshold lands near risk~55-65.
_W = {
    "ttc": 2.4,
    "dcpa": 2.0,
    "conflict": 1.5,
    "closing": 1.1,
    "intent": 0.9,
    "veh_speed": 0.8,
    "turn": 0.7,
    "on_road": 0.7,
    "not_looking": 0.5,
}
_BIAS = -2.4
_BRAKE_RELIEF = 1.1

# human-readable labels for the explanation panel
FACTOR_LABELS = {
    "ttc": "Low time-to-collision",
    "dcpa": "Small predicted miss distance",
    "conflict": "Trajectory intersection",
    "closing": "High closing speed",
    "intent": "Pedestrian crossing intent",
    "veh_speed": "Vehicle speed",
    "turn": "Vehicle turning across path",
    "on_road": "Pedestrian in roadway",
    "not_looking": "Pedestrian not looking",
    "brake_relief": "Vehicle braking (mitigating)",
}


def _factors(f: dict) -> dict:
    """Map raw interaction features to bounded [0,1] danger factors."""
    nm = RISK.nearmiss_dcpa
    return {
        "ttc": float(np.clip(f.get("inv_ttc", 0.0) / (1.0 / 0.5), 0, 1)),          # ttc<=0.5s -> 1
        "dcpa": float(np.clip((2 * nm - f.get("dcpa", 99.0)) / (2 * nm), 0, 1)),   # dcpa 0 -> 1
        "conflict": float(np.clip(f.get("path_conflict", 0.0), 0, 1)),
        "closing": float(np.clip(f.get("closing_speed", 0.0) / 10.0, 0, 1)),
        "intent": float(np.clip(f.get("vru_intention", 0.0), 0, 1)),
        "veh_speed": float(np.clip(f.get("veh_speed", 0.0) / 13.0, 0, 1)),
        "turn": float(f.get("veh_turning", 0.0)),
        "on_road": float(f.get("vru_on_road", 0.0)),
        "not_looking": float(f.get("vru_not_looking", 0.0)),
    }


class PhysicsRisk:
    """Interpretable logistic risk with built-in factor attribution."""

    def __init__(self, scale: float = 1.0, bias_shift: float = 0.0):
        self.scale = scale            # calibration gain on the logit
        self.bias_shift = bias_shift

    def score(self, features: dict) -> tuple[float, dict]:
        fac = _factors(features)
        terms = {k: _W[k] * fac[k] for k in _W}
        brake = _BRAKE_RELIEF * float(features.get("veh_braking", 0.0))
        logit = _BIAS + self.bias_shift + sum(terms.values()) - brake
        logit *= self.scale
        p = 1.0 / (1.0 + math.exp(-logit))
        risk = 100.0 * p
        # contributions: share of the *positive* drive, in risk points
        pos = {FACTOR_LABELS[k]: v for k, v in terms.items() if v > 1e-3}
        total = sum(pos.values())
        contrib = {}
        if total > 0:
            for k, v in pos.items():
                contrib[k] = round(risk * v / total, 1)
        if brake > 1e-3:
            contrib[FACTOR_LABELS["brake_relief"]] = round(-min(risk, 100 * brake / (total + 1e-6)), 1)
        return float(risk), contrib


class LearnedRisk:
    """Calibrated gradient-boosted risk model (loaded from disk)."""

    def __init__(self, model, calibrator=None):
        self.model = model
        self.calibrator = calibrator

    @staticmethod
    def load(path=None):
        path = path or (MODELS_DIR / "risk_model.pkl")
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return LearnedRisk(d["model"], d.get("calibrator"))

    def proba(self, features: dict) -> float:
        return float(self.proba_batch([features])[0])

    def proba_batch(self, feature_dicts) -> np.ndarray:
        """Score many interactions in one model call, essential for pipeline throughput
        (sklearn's per-call overhead makes one-at-a-time scoring ~10x slower)."""
        if not feature_dicts:
            return np.zeros(0)
        X = np.array([[f.get(k, 0.0) for k in RISK_FEATURES] for f in feature_dicts], float)
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)[:, 1]
        else:
            p = self.model.predict(X)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)


class RiskEngine:
    """Scores interactions; prefers the learned model, always attaches physics
    contributions for the explanation panel."""

    def __init__(self, use_learned: bool = True):
        self.physics = PhysicsRisk()
        self.learned = LearnedRisk.load() if use_learned else None

    @property
    def is_learned(self) -> bool:
        return self.learned is not None

    def band(self, risk: float) -> RiskBand:
        if risk >= RISK.band_critical:
            return RiskBand.CRITICAL
        if risk >= RISK.band_high:
            return RiskBand.HIGH
        if risk >= RISK.band_medium:
            return RiskBand.MEDIUM
        return RiskBand.LOW

    def score_interaction(self, inter: Interaction) -> Interaction:
        phys_risk, contrib = self.physics.score(inter.features)
        risk = 100.0 * self.learned.proba(inter.features) if self.learned is not None else phys_risk
        inter.risk_score = float(risk)
        inter.band = self.band(risk)
        inter.contributions = contrib
        return inter

    def score_all(self, interactions) -> None:
        if not interactions:
            return
        # physics contributions (cheap, always needed for the explanation panel)
        contribs = [self.physics.score(it.features) for it in interactions]
        if self.learned is not None:
            risks = 100.0 * self.learned.proba_batch([it.features for it in interactions])
        else:
            risks = [pr for pr, _ in contribs]
        for it, (pr, contrib), risk in zip(interactions, contribs, risks):
            it.risk_score = float(risk)
            it.band = self.band(float(risk))
            it.contributions = contrib
