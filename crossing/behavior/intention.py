"""Crossing-intention estimation and behaviour classification.

The interesting scientific question in pedestrian safety is not *where is the person* but
*will they step into the road: before they do it?* CROSSING answers it as a calibrated
probability that evolves over time (0.21 -> 0.37 -> 0.68 -> 0.94), never a hard boolean.

Two estimators share one interface:

* a **heuristic** built from kinematics + context (distance to curb, motion toward the
  road, signal state, gap to traffic), always available, no training;
* a **learned GRU** (``crossing.ml.nn``) over the last ~1.5 s of behavioural features,
  trained on labelled crossings, and used when a model file is present.

Both return ``(p_cross, std)``; uncertainty is the Bernoulli spread inflated by how much
of the observation window was actually visible (occlusion => less confident), which is the
honest thing to do in a safety system.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import MODELS_DIR
from ..schema import Behavior, Attention, EntityState
from .features import N_FEATURES

OBS_FRAMES = 45          # 1.5 s history at 30 fps
SEQ_T = 15               # subsampled sequence length fed to the GRU
STRIDE = 3


def build_sequence(history: list[np.ndarray], T: int = SEQ_T, stride: int = STRIDE) -> np.ndarray:
    """Turn a list of per-frame feature vectors into a fixed (T, N) array.

    Takes the most recent ``T*stride`` frames, subsamples by ``stride``, and front-pads
    with the earliest available vector when history is short.
    """
    if not history:
        return np.zeros((T, N_FEATURES))
    window = history[-(T * stride):]
    sub = window[::stride][-T:]
    if len(sub) < T:
        pad = [sub[0]] * (T - len(sub))
        sub = pad + list(sub)
    return np.array(sub, dtype=float)


def classify_behavior(state: EntityState, dist_curb: float | None = None) -> Behavior:
    """Coarse pedestrian behaviour from kinematics + context."""
    sp = state.speed
    on_road = dist_curb is not None and dist_curb < 0.2
    if on_road and sp > 0.4:
        return Behavior.RUNNING if sp > 2.4 else Behavior.CROSSING
    if sp < 0.25:
        # standing at a curb we call "waiting"; elsewhere "standing"
        if dist_curb is not None and 0 <= dist_curb < 1.5:
            return Behavior.WAITING
        return Behavior.STANDING
    if state.jerk > 6.0 and sp > 0.4:
        return Behavior.TURNING
    return Behavior.WALKING


def classify_attention(state: EntityState) -> Attention:
    gaze = state.attrs.get("gaze_cue")
    look = state.attrs.get("looking_score")
    if look is None and gaze is None:
        return Attention.UNCERTAIN
    score = look if look is not None else (0.5 + 0.5 * abs(gaze or 0.0))
    if score > 0.6:
        return Attention.LOOKING
    if score < 0.4:
        return Attention.NOT_LOOKING
    return Attention.UNCERTAIN


def heuristic_intention(seq: np.ndarray, visibility: float = 1.0) -> tuple[float, float]:
    """A transparent baseline P(cross) from the latest feature frame + short trend.

    Uses feature indices from :data:`crossing.behavior.features.FEATURE_NAMES`.
    """
    last = seq[-1]
    speed, accel, dist_curb, on_road, lat_prog, head_align, gaze, ped_walk, veh_dist, veh_ttc, veh_speed = last
    # base drive: proximity to curb + motion toward road + heading aligned with crossing
    approach = np.clip(1.0 - abs(dist_curb) / 1.0, 0, 1)     # dist_curb already /5
    logit = -1.2
    logit += 2.6 * on_road
    logit += 1.8 * approach * head_align
    logit += 1.4 * max(0.0, speed) * head_align
    logit += 1.1 * ped_walk                                  # WALK encourages, DONT_WALK discourages
    logit += 0.8 * (lat_prog > 0.05 and lat_prog < 0.95)     # mid-crossing
    logit -= 0.6 * veh_ttc                                   # imminent car -> may hold back
    # rising trend in on_road / speed over the window nudges intention up
    trend = float(seq[-1][0] - seq[0][0]) + float(seq[-1][3] - seq[0][3])
    logit += 1.2 * np.clip(trend, -1, 1)
    p = 1.0 / (1.0 + math.exp(-logit))
    std = math.sqrt(p * (1 - p)) * (0.5 + 0.5 * (1 - visibility))
    return float(p), float(std)


class IntentionModel:
    """Unified interface over the heuristic and the learned GRU."""

    def __init__(self, use_learned: bool = True):
        self.gru = None
        if use_learned:
            path = MODELS_DIR / "intention_gru.npz"
            if path.exists():
                try:
                    from ..ml.nn import GRU
                    self.gru = GRU.load(str(path))
                except Exception:
                    self.gru = None

    @property
    def is_learned(self) -> bool:
        return self.gru is not None

    def predict(self, seq: np.ndarray, visibility: float = 1.0) -> tuple[float, float]:
        if self.gru is not None:
            p = float(self.gru.predict_proba(seq[None, :, :])[0])
            # Bernoulli spread inflated by occlusion
            std = math.sqrt(p * (1 - p)) * (0.4 + 0.6 * (1 - visibility))
            return p, float(std)
        return heuristic_intention(seq, visibility)
