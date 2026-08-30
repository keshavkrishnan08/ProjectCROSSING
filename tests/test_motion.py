"""A vehicle that is not moving must never raise a pedestrian's risk, however close it looks.

These lock the motion gate in ``hit_probability``: the same close, bearing-down geometry scores
high when the vehicle is moving and exactly zero when it is not, and Kalman-velocity jitter on a
stationary vehicle cannot leak through.
"""
from __future__ import annotations

from scripts.safety_video import hit_probability


def _ped():
    return {"id": 1, "cls": "person", "box": [620, 560, 660, 700], "vx": 0.0, "vy": 0.0}


def _car(moving, vx=-4.0, vy=2.0):
    # a car right in front of the pedestrian, heading toward them
    return {"id": 2, "cls": "car", "box": [600, 540, 720, 690], "vx": vx, "vy": vy, "moving": moving}


def test_moving_car_bearing_down_is_high_risk():
    assert hit_probability(_ped(), [_car(True)], fps=30, frame_w=1280) > 0.7


def test_stationary_car_is_zero_risk_even_when_close():
    assert hit_probability(_ped(), [_car(False)], fps=30, frame_w=1280) == 0.0


def test_kalman_jitter_on_a_stationary_car_does_not_leak_through():
    jittering = _car(False, vx=1.6, vy=-1.3)   # nonzero velocity, but not actually moving
    assert hit_probability(_ped(), [jittering], fps=30, frame_w=1280) == 0.0


def test_moving_flag_absent_defaults_to_scoring():
    car = _car(True)
    del car["moving"]                          # backward-compatible default
    assert hit_probability(_ped(), [car], fps=30, frame_w=1280) > 0.7
