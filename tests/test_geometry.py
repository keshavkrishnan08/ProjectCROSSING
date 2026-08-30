"""Conflict geometry + homography: closed‑form checks against hand‑computed cases."""
import math

import numpy as np

from crossing.interaction import metrics as M
from crossing.worldmodel.homography import Homography, synthetic_camera_homography


def test_ttc_head_on():
    # gap 10, closing 2 m/s, footprint radius 1 -> touch at (10-1)/2 = 4.5 s
    assert abs(M.ttc((0, 0), (1, 0), (10, 0), (-1, 0), 1.0) - 4.5) < 1e-6


def test_cpa_head_on():
    dcpa, tcpa = M.cpa((0, 0), (1, 0), (10, 0), (-1, 0))
    assert dcpa < 1e-6 and abs(tcpa - 5.0) < 1e-6


def test_passing_miss_has_infinite_ttc():
    assert math.isinf(M.ttc((0, 0), (1, 0), (10, 2), (-1, 0), 1.0))
    assert abs(M.cpa((0, 0), (1, 0), (10, 2), (-1, 0))[0] - 2.0) < 1e-6


def test_diverging():
    assert math.isinf(M.ttc((0, 0), (-1, 0), (10, 0), (1, 0), 1.0))
    assert M.closing_speed((0, 0), (-1, 0), (10, 0), (1, 0)) < 0


def test_perpendicular_pet_zero():
    # both reach the crossing point (10,0) at t=5 -> PET 0
    assert abs(M.pet((0, 0), (2, 0), (10, -10), (0, 2))) < 1e-6


def test_closing_speed_sign():
    assert M.closing_speed((0, 0), (1, 0), (5, 0), (0, 0)) > 0     # approaching
    assert M.closing_speed((0, 0), (-1, 0), (5, 0), (0, 0)) < 0    # receding


def test_homography_roundtrip_and_dlt():
    H = synthetic_camera_homography(1920, 1080)
    gpts = [[-2, 0], [42, 0], [42, 46], [-2, 46], [20, 23], [10, 5]]
    img = H.ground_to_image(gpts)
    back = H.image_to_ground(img)
    assert np.abs(np.array(gpts) - back).max() < 1e-8
    # DLT re‑estimation from correspondences recovers the same map
    H2 = Homography.fit(img, gpts)
    assert np.abs(np.array(gpts) - H2.image_to_ground(img)).max() < 1e-8


def test_homography_frames_scene_in_image():
    H = synthetic_camera_homography(1920, 1080)
    for g in [[-2, 0], [42, 0], [42, 46], [-2, 46], [20, 23]]:
        u, v = H.point_ground_to_image(*g)
        assert -5 <= u <= 1925 and -5 <= v <= 1085
