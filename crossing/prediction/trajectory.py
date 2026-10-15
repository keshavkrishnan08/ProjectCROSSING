"""Multi-modal trajectory prediction.

The future is not a single line. A pedestrian approaching the kerb might continue across,
stop, or turn back; the system should say so with probabilities and draw each hypothesis.
CROSSING predicts several futures per entity, each a sequence of ``[t, x, y]`` waypoints
with growing positional uncertainty, and weights them by the current kinematics and the
estimated crossing intention.

* **Pedestrians** get up to three goal-conditioned modes, *continue*, *stop*, *turn back*
, whose probabilities are driven by intention, motion, and signal context.
* **Vehicles** get a dominant kinematic mode (constant-acceleration along the lane) plus a
  braking mode when a conflict is developing.

The predictor is deliberately transparent physics; an optional learned residual (a GRU on
the same behavioural features) can refine it when trained, but is never required.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Horizons, HORIZONS, COMFORT_DECEL, MAX_DECEL
from ..schema import EntityState, EntityKind, Prediction, TrajectoryMode


def _sample_times(horizon: float, step: float = 0.5) -> np.ndarray:
    n = max(1, int(round(horizon / step)))
    return np.linspace(step, horizon, n)


def _ca_rollout(p0, v0, a0, times, damp_accel: float = 0.6):
    """Constant-acceleration rollout with acceleration damped over time (accel decays)."""
    pts = []
    for t in times:
        decay = math.exp(-t / 1.2)
        a = a0 * (damp_accel * decay)
        pos = p0 + v0 * t + 0.5 * a * t * t
        pts.append([float(t), float(pos[0]), float(pos[1])])
    return pts


def _stop_rollout(p0, v0, times, decel=COMFORT_DECEL):
    """Decelerate along current heading to a stop."""
    speed = np.linalg.norm(v0)
    if speed < 1e-3:
        return [[float(t), float(p0[0]), float(p0[1])] for t in times]
    hat = v0 / speed
    t_stop = speed / decel
    pts = []
    for t in times:
        if t < t_stop:
            d = speed * t - 0.5 * decel * t * t
        else:
            d = 0.5 * speed * t_stop
        pos = p0 + hat * d
        pts.append([float(t), float(pos[0]), float(pos[1])])
    return pts


def _turn_back_rollout(p0, v0, times, speed=1.1):
    sp = np.linalg.norm(v0)
    hat = (-v0 / sp) if sp > 1e-3 else np.array([0.0, 0.0])
    pts = []
    for t in times:
        # brief deceleration then reverse
        d = -0.3 * sp * min(t, 0.5) + speed * max(0.0, t - 0.5)
        pos = p0 + hat * d
        pts.append([float(t), float(pos[0]), float(pos[1])])
    return pts


def _std_profile(times, base=0.25, growth=0.35):
    return [round(base + growth * t, 3) for t in times]


class TrajectoryPredictor:
    def __init__(self, horizons: Horizons = HORIZONS, step: float = 0.5):
        self.h = horizons
        self.step = step
        self.times = _sample_times(horizons.predict_max, step)

    def predict(self, e: EntityState) -> Prediction:
        if e.kind == EntityKind.PERSON:
            modes = self._predict_ped(e)
        elif e.kind == EntityKind.CYCLIST:
            modes = self._predict_vehicle(e, ped_like=True)
        else:
            modes = self._predict_vehicle(e)
        return Prediction(entity_id=e.id, frame=e.frame, modes=modes)

    # -- pedestrians ------------------------------------------------------------------
    def _predict_ped(self, e: EntityState):
        p0 = np.array([e.position.x, e.position.y])
        v0 = np.array([e.velocity.x, e.velocity.y])
        a0 = np.array([e.acceleration.x, e.acceleration.y])
        intent = float(e.crossing_intention)
        on_road = float(e.attrs.get("on_road", 0.0)) > 0.5
        speed = e.speed

        std = _std_profile(self.times)
        modes = []
        # mode weights from intention + motion
        w_cont = 0.25 + 0.7 * intent + 0.3 * min(speed / 1.5, 1.0)
        w_stop = 0.25 + 0.7 * (1 - intent) + (0.3 if not on_road else -0.1)
        w_back = 0.12 * (1 - intent) * (1 - on_road)
        w = np.array([w_cont, max(0.05, w_stop), max(0.0, w_back)])
        w = w / w.sum()

        modes.append(TrajectoryMode(float(w[0]), "continues",
                                    _ca_rollout(p0, v0, a0, self.times), std))
        modes.append(TrajectoryMode(float(w[1]), "stops",
                                    _stop_rollout(p0, v0, self.times), std))
        if w[2] > 0.03:
            modes.append(TrajectoryMode(float(w[2]), "turns back",
                                        _turn_back_rollout(p0, v0, self.times), std))
        # renormalise if the turn-back mode was dropped
        s = sum(m.prob for m in modes)
        for m in modes:
            m.prob = round(m.prob / s, 4)
        return sorted(modes, key=lambda m: -m.prob)

    # -- vehicles / cyclists ----------------------------------------------------------
    def _predict_vehicle(self, e: EntityState, ped_like: bool = False):
        p0 = np.array([e.position.x, e.position.y])
        v0 = np.array([e.velocity.x, e.velocity.y])
        a0 = np.array([e.acceleration.x, e.acceleration.y])
        std = _std_profile(self.times, base=0.3, growth=0.45)
        modes = [TrajectoryMode(0.82, "maintains",
                                _ca_rollout(p0, v0, a0, self.times, damp_accel=0.9), std)]
        # a braking alternative (relevant when a conflict is developing)
        if e.speed > 2.0:
            modes.append(TrajectoryMode(0.18, "brakes",
                                        _stop_rollout(p0, v0, self.times,
                                                      decel=min(MAX_DECEL, COMFORT_DECEL * 1.6)), std))
        s = sum(m.prob for m in modes)
        for m in modes:
            m.prob = round(m.prob / s, 4)
        return modes
