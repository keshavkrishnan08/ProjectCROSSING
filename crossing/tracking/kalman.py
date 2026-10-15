"""Kalman filters used across CROSSING.

Two are provided:

* :class:`BBoxKalman`: the classic SORT/ByteTrack 8-state constant-velocity filter on
  ``[cx, cy, a, h]`` (centre, aspect ratio, height) used by the tracker to smooth boxes
  and gate associations with Mahalanobis distance.
* :class:`GroundKalman`: a 6-state constant-acceleration filter on ``[x, y, vx, vy, ax,
  ay]`` in ground metres, used by the world model to produce stable velocity /
  acceleration / heading / jerk from noisy projected foot points.

Both are plain NumPy; no filterpy dependency.
"""
from __future__ import annotations

import numpy as np


class BBoxKalman:
    """8-state constant-velocity Kalman filter on bounding boxes (SORT convention)."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, meas: np.ndarray):
        mean_pos = meas
        mean_vel = np.zeros(4)
        mean = np.r_[mean_pos, mean_vel]
        std = np.array([
            2 * self._std_pos * meas[3], 2 * self._std_pos * meas[3], 1e-2, 2 * self._std_pos * meas[3],
            10 * self._std_vel * meas[3], 10 * self._std_vel * meas[3], 1e-5, 10 * self._std_vel * meas[3],
        ])
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean, cov):
        std = np.array([
            self._std_pos * mean[3], self._std_pos * mean[3], 1e-2, self._std_pos * mean[3],
            self._std_vel * mean[3], self._std_vel * mean[3], 1e-5, self._std_vel * mean[3],
        ])
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def project(self, mean, cov):
        std = np.array([self._std_pos * mean[3], self._std_pos * mean[3], 1e-1, self._std_pos * mean[3]])
        R = np.diag(np.square(std))
        pm = self._H @ mean
        pc = self._H @ cov @ self._H.T + R
        return pm, pc

    def update(self, mean, cov, meas):
        pm, pc = self.project(mean, cov)
        K = cov @ self._H.T @ np.linalg.inv(pc)
        mean = mean + K @ (meas - pm)
        cov = cov - K @ pc @ K.T
        return mean, cov

    def gating_distance(self, mean, cov, measurements) -> np.ndarray:
        pm, pc = self.project(mean, cov)
        d = np.atleast_2d(measurements) - pm
        try:
            L = np.linalg.cholesky(pc)
            z = np.linalg.solve(L, d.T)
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            return np.sum(d @ np.linalg.pinv(pc) * d, axis=1)


class GroundKalman:
    """6-state constant-acceleration Kalman filter in ground metres.

    State ``[x, y, vx, vy, ax, ay]``. Produces smooth kinematics from the noisy
    homography-projected ground positions of a tracked entity.
    """

    def __init__(self, dt: float, q: float = 2.5, r: float = 0.25):
        self.dt = dt
        dt2 = 0.5 * dt * dt
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], float)
        self.H = np.zeros((2, 6)); self.H[0, 0] = 1; self.H[1, 1] = 1
        # Process noise: white-jerk model, block-diagonal over the two independent axes.
        # State order is [x, y, vx, vy, ax, ay], so the x-states are indices {0,2,4} and
        # the y-states {1,3,5}; each axis gets its own [dt^2/2, dt, 1] noise block, with no
        # cross-axis correlation (a single outer(G,G) would wrongly couple x and y).
        g = np.array([dt2, dt, 1.0])
        Qax = np.outer(g, g) * q ** 2
        Q = np.zeros((6, 6))
        xi, yi = (0, 2, 4), (1, 3, 5)
        for a in range(3):
            for b in range(3):
                Q[xi[a], xi[b]] = Qax[a, b]
                Q[yi[a], yi[b]] = Qax[a, b]
        self.Q = Q
        self.R = np.eye(2) * r ** 2
        self.x = None
        self.P = None

    def initiate(self, pos):
        self.x = np.array([pos[0], pos[1], 0, 0, 0, 0], float)
        self.P = np.diag([0.5, 0.5, 4.0, 4.0, 4.0, 4.0]).astype(float)
        return self

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, pos):
        z = np.asarray(pos, float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[2:4].copy()

    @property
    def acceleration(self):
        return self.x[4:6].copy()
