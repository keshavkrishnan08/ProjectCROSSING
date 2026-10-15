"""Pose-derived cues for pedestrians.

Detecting *where* a person is is not enough; safety depends on what they are *doing* 
walking vs standing vs turning, and crucially whether they are **looking at traffic**. A
pedestrian scanning the road before stepping off behaves very differently from a
head-down phone user. This module turns keypoints (from YOLO-Pose, or the simulator's
synthetic skeleton) into two scalar cues consumed by the behaviour model:

* ``gaze_lateral``: signed head offset relative to the shoulders (a proxy for whether the
  head is turned toward oncoming traffic);
* ``upright`` / ``stance``: coarse body configuration.

Keypoint order follows the simulator/COCO-lite convention:
``[nose, L-shoulder, R-shoulder, L-hip, R-hip]``.
"""
from __future__ import annotations

import numpy as np


def gaze_lateral(keypoints) -> float:
    """Head lateral offset normalised by shoulder width; +ve = head turned right.

    Returns 0 when keypoints are missing/low-confidence.
    """
    if not keypoints or len(keypoints) < 3:
        return 0.0
    nose = keypoints[0]
    lsh, rsh = keypoints[1], keypoints[2]
    if min(nose[2], lsh[2], rsh[2]) < 0.2:
        return 0.0
    shoulder_mid_x = 0.5 * (lsh[0] + rsh[0])
    width = abs(rsh[0] - lsh[0]) + 1e-6
    return float((nose[0] - shoulder_mid_x) / width)


def looking_score(keypoints, cross_axis_sign: float) -> float:
    """Heuristic P(looking toward traffic) in [0,1].

    ``cross_axis_sign`` encodes which way oncoming traffic lies relative to the person's
    motion; a head turned that way raises the score. With no keypoints, returns 0.5
    (uncertain), which the behaviour model treats as low-information.
    """
    if not keypoints:
        return 0.5
    g = gaze_lateral(keypoints)
    # a decisive head turn (|g| large) toward the traffic side -> higher looking score
    aligned = g * cross_axis_sign
    return float(np.clip(0.5 + 1.6 * aligned, 0.0, 1.0))


class PoseEstimator:  # pragma: no cover - optional real-model path
    """Optional YOLO-Pose wrapper; returns keypoints per detection when available."""

    def __init__(self, model_name: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_name)

    def estimate(self, frame):
        res = self.model.predict(frame, verbose=False)[0]
        out = []
        if res.keypoints is not None:
            for kp in res.keypoints.data:
                out.append([[float(x), float(y), float(c)] for x, y, c in kp.tolist()])
        return out
