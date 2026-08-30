"""Per-entity kinematic estimation in ground space.

A tracked box gives a noisy ground foot-point each frame. :class:`GroundTrack` turns that
stream into stable **position, velocity, acceleration, heading, speed and jerk** via a
constant-acceleration Kalman filter, plus lightly-smoothed heading (headings are noisy at
low speed, so we hold the last confident heading when nearly stationary).

Richer than a bounding box: this is what every physical calculation downstream consumes.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..tracking.kalman import GroundKalman


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class GroundTrack:
    """Kinematic history + Kalman state for one entity on the ground plane."""

    def __init__(self, track_id: int, dt: float, is_vehicle: bool):
        self.id = track_id
        self.dt = dt
        # vehicles are heavier/smoother -> lower process noise; peds more agile.
        # r is matched to the real projected foot-point noise (~0.5 m for vehicles) so the
        # filter smooths velocity/heading instead of chasing pixel jitter.
        q = 0.9 if is_vehicle else 2.6
        r = 0.45 if is_vehicle else 0.30
        self.is_vehicle = is_vehicle
        self.kf = GroundKalman(dt=dt, q=q, r=r)
        self.initialised = False
        self.heading = 0.0
        self._last_acc = np.zeros(2)
        self.jerk = 0.0
        self.n = 0
        self.pos_hist = deque(maxlen=120)
        self.vel_hist = deque(maxlen=120)
        self.t_hist = deque(maxlen=120)
        self.missed = 0

    def update(self, pos, t: float, observed: bool = True):
        pos = np.asarray(pos, float)
        if not self.initialised:
            self.kf.initiate(pos)
            self.initialised = True
        else:
            self.kf.predict()
            if observed:
                self.kf.update(pos)
        self.n += 1
        self.missed = 0 if observed else self.missed + 1

        v = self.kf.velocity
        a = self.kf.acceleration
        self.jerk = float(np.linalg.norm(a - self._last_acc) / self.dt)
        self._last_acc = a.copy()

        # Heading: only trust the velocity direction once clearly moving, and low-pass it
        # (shortest-angle) so slow/queued actors don't jitter or point off-road. Below the
        # threshold we hold the last confident heading.
        speed = float(np.linalg.norm(v))
        thresh = 0.7 if self.is_vehicle else 0.4
        if speed > thresh:
            target = math.atan2(v[1], v[0])
            alpha = min(0.5, 0.12 + 0.10 * speed)   # faster -> trust new heading more
            self.heading = self.heading + wrap_angle(target - self.heading) * alpha
        self.pos_hist.append(self.kf.position.copy())
        self.vel_hist.append(v.copy())
        self.t_hist.append(t)
        return self

    def coast(self, t: float):
        """Advance the filter with no measurement (occluded frame)."""
        if self.initialised:
            self.update(self.kf.position, t, observed=False)
        return self

    @property
    def position(self):
        return self.kf.position

    @property
    def velocity(self):
        return self.kf.velocity

    @property
    def acceleration(self):
        return self.kf.acceleration

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.kf.velocity))

    @property
    def accel_mag(self) -> float:
        return float(np.linalg.norm(self.kf.acceleration))

    def recent_positions(self, k: int) -> np.ndarray:
        arr = list(self.pos_hist)[-k:]
        return np.array(arr) if arr else np.zeros((0, 2))

    def history_array(self, k: int):
        """Return (positions[k,2], velocities[k,2], times[k]) padded at the front."""
        pos = np.array(list(self.pos_hist)[-k:]) if self.pos_hist else np.zeros((0, 2))
        vel = np.array(list(self.vel_hist)[-k:]) if self.vel_hist else np.zeros((0, 2))
        ts = np.array(list(self.t_hist)[-k:]) if self.t_hist else np.zeros((0,))
        return pos, vel, ts
