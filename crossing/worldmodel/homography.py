"""Image <-> ground-plane geometry.

Pixel coordinates are not physically useful: two pedestrians the same distance from a
car can be hundreds of pixels apart or nearly overlapping depending on depth. Every
physical quantity in CROSSING (velocity, TTC, DCPA, risk) is computed in **ground
metres**, so the first thing the world model does is lift each detection's ground-contact
point onto the road plane with a homography.

This module provides:

* :class:`Homography`: a 3x3 projective map with forward/inverse application and a
  Direct-Linear-Transform estimator (``fit``) from >=4 correspondences, exactly how you
  would calibrate a real fixed camera from lane-marking / crosswalk corners.
* :func:`synthetic_camera_homography`: build a physically-correct ground->image
  homography from a pinhole camera (height, pitch, focal length). The simulator uses this
  to render, and the pipeline re-estimates it from known scene points, closing the loop.
"""
from __future__ import annotations

import math

import numpy as np


class Homography:
    """A 3x3 projective transform between image pixels and ground metres."""

    def __init__(self, H_img2ground: np.ndarray):
        H = np.asarray(H_img2ground, dtype=float).reshape(3, 3)
        H = H / H[2, 2]
        self.H = H
        self.Hinv = np.linalg.inv(H)

    # -- application -------------------------------------------------------------------
    def image_to_ground(self, pts) -> np.ndarray:
        """(N,2) pixels -> (N,2) ground metres."""
        return _apply(self.H, np.atleast_2d(pts))

    def ground_to_image(self, pts) -> np.ndarray:
        """(N,2) ground metres -> (N,2) pixels."""
        return _apply(self.Hinv, np.atleast_2d(pts))

    def point_image_to_ground(self, x: float, y: float) -> tuple[float, float]:
        p = self.image_to_ground([[x, y]])[0]
        return float(p[0]), float(p[1])

    def point_ground_to_image(self, x: float, y: float) -> tuple[float, float]:
        p = self.ground_to_image([[x, y]])[0]
        return float(p[0]), float(p[1])

    def to_list(self) -> list[list[float]]:
        return self.H.tolist()

    # -- calibration -------------------------------------------------------------------
    @staticmethod
    def fit(image_pts, ground_pts) -> "Homography":
        """Estimate image->ground homography from >=4 correspondences via normalized DLT."""
        img = np.asarray(image_pts, float)
        gnd = np.asarray(ground_pts, float)
        if len(img) < 4:
            raise ValueError("need >= 4 correspondences")
        Ti, imgn = _normalize(img)
        Tg, gndn = _normalize(gnd)
        A = []
        for (u, v), (X, Y) in zip(imgn, gndn):
            A.append([-u, -v, -1, 0, 0, 0, u * X, v * X, X])
            A.append([0, 0, 0, -u, -v, -1, u * Y, v * Y, Y])
        A = np.asarray(A)
        _, _, Vt = np.linalg.svd(A)
        Hn = Vt[-1].reshape(3, 3)
        H = np.linalg.inv(Tg) @ Hn @ Ti
        return Homography(H)

    @staticmethod
    def from_list(m) -> "Homography":
        return Homography(np.asarray(m, float))


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = (M @ h.T).T
    out = out[:, :2] / out[:, 2:3]
    return out


def _normalize(pts: np.ndarray):
    """Isotropic normalization (Hartley) for a well-conditioned DLT."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2) / d if d > 1e-9 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ptsn = _apply(T, pts)
    return T, ptsn


# --------------------------------------------------------------------------------------
# Pinhole camera -> planar homography
# --------------------------------------------------------------------------------------
def synthetic_camera_homography(
    image_w: int,
    image_h: int,
    ground_bounds: tuple[float, float, float, float] = (-2.0, 0.0, 42.0, 46.0),
    horizon_frac: float = 0.16,
    near_margin_frac: float = 0.02,
    far_converge: float = 0.30,
    bottom_frac: float = 0.99,
) -> Homography:
    """Build a realistic elevated-street-cam homography that frames ``ground_bounds``.

    Rather than juggle extrinsics, we do what a calibration technician does: pin four
    known ground points (the corners of the region of interest) to four image points
    forming a perspective trapezoid, then fit the homography. Far ground rows (large Y)
    converge toward a vanishing band near the top; near rows span the bottom. This
    guarantees the scene lands inside the frame with correct projective foreshortening.

    Returns the *image->ground* :class:`Homography`.
    """
    xmin, ymin, xmax, ymax = ground_bounds
    W, H = float(image_w), float(image_h)

    top_y = horizon_frac * H
    bot_y = bottom_frac * H
    cx = W / 2.0
    near_half = (0.5 - near_margin_frac) * W
    far_half = far_converge * near_half

    # ground quad: near edge (ymin) wide at bottom, far edge (ymax) narrow near horizon.
    ground_quad = [
        [xmin, ymin],  # near-left
        [xmax, ymin],  # near-right
        [xmax, ymax],  # far-right
        [xmin, ymax],  # far-left
    ]
    image_quad = [
        [cx - near_half, bot_y],   # near-left  -> bottom-left
        [cx + near_half, bot_y],   # near-right -> bottom-right
        [cx + far_half, top_y],    # far-right  -> top-right
        [cx - far_half, top_y],    # far-left   -> top-left
    ]
    return Homography.fit(image_quad, ground_quad)
