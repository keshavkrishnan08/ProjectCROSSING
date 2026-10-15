"""Probability that a pedestrian is hit, from depth-aware box geometry and tracked motion.

Pure geometry with no video, model, or asset dependencies, so it can be unit-tested and reused
outside the video script (``scripts/safety_video.py``).
"""
from __future__ import annotations

import math

import numpy as np


def foot_point(b):
    """Bottom-centre of a box, where the actor meets the ground."""
    return (0.5 * (b[0] + b[2]), b[3])


# typical real heights (metres) used to turn a box into an approximate depth
REAL_HEIGHT_M = {"person": 1.7, "bicycle": 1.7, "car": 1.5, "van": 2.1, "truck": 3.0,
                 "bus": 3.2, "motorbike": 1.5, "motorcycle": 1.5}


def ground_position(item, f, cx):
    """Approximate ground position (lateral X, depth Z) in metres from a box.

    A pinhole camera images an object of real height H at depth Z with pixel height
    h = f*H/Z, so Z = f*H/h. Its lateral offset is X = (foot_x - cx) * Z / f. This lets us
    recover a metric distance that is robust to the '2D boxes look close but are far apart in
    depth' illusion the raw image gives.
    """
    b = item["box"]
    fx = 0.5 * (b[0] + b[2])
    h = max(b[3] - b[1], 6.0)
    z = f * REAL_HEIGHT_M.get(item["cls"], 1.6) / h
    x = (fx - cx) * z / f
    return x, z


def hit_probability(ped, vehicles, fps=30.0, frame_w=1280.0, explain=False):
    """Estimate P(pedestrian is hit) from REAL (depth-aware) geometry + tracked motion.

    The core fix over naive image proximity: recover an approximate metric position for every
    actor from its box (height -> depth, horizontal offset -> lateral position), so a car that
    merely overlaps a pedestrian in the 2D frame but sits far behind or ahead of them scores
    near zero. On top of the real distance we still require the vehicle to be moving and headed
    toward the person, and we ramp risk with the resulting time-to-collision.

    With ``explain=True`` also returns the winning vehicle and the metric components behind the
    score, so the value can be traced back to its inputs (see :mod:`crossing.provenance`).
    """
    f = 0.9 * frame_w
    cx = 0.5 * frame_w
    Xp, Zp = ground_position(ped, f, cx)
    ph = max(ped["box"][3] - ped["box"][1], 12.0)
    pvx, pvy = ped["vx"], ped["vy"]
    best = 0.0
    best_detail = None
    for v in vehicles:
        # A vehicle that is not actually moving is never a threat, however close it looks.
        # ``moving`` is a robust flag from real displacement over recent frames (set in the
        # processing loop); it defeats Kalman velocity jitter that a parked car can show.
        if not v.get("moving", True):
            continue
        Xv, Zv = ground_position(v, f, cx)
        d_real = math.hypot(Xp - Xv, Zp - Zv)          # metres, depth-aware
        if d_real > 9.0:                               # genuinely far apart in 3D
            continue
        vh = max(v["box"][3] - v["box"][1], 12.0)
        scale = 0.5 * (ph + vh)
        vvx, vvy = v["vx"], v["vy"]
        v_speed = math.hypot(vvx, vvy)
        if v_speed / scale < 0.006:                    # essentially parked/stationary vehicle
            continue
        px, py = foot_point(ped["box"]); vx, vy = foot_point(v["box"])
        dx, dy = px - vx, py - vy
        gap = math.hypot(dx, dy) + 1e-6
        align = (vvx * dx + vvy * dy) / (v_speed * gap)    # 1 = heading straight at ped
        if align < 0.15:                               # moving away or across -> not a threat
            continue
        # Close + squarely in the vehicle's path + moving is dangerous even at a modest tracked
        # speed (a car covers 3 m in well under a second), so proximity and path drive the base
        # risk; an actively-closing vehicle (fast, short TTC) amplifies it further.
        prox = float(np.clip((6.5 - d_real) / 5.5, 0, 1))  # 1 by ~1 m, 0 by 6.5 m
        path = float(np.clip((align - 0.1) / 0.7, 0, 1))
        closing_px = (dx * (vvx - pvx) + dy * (vvy - pvy)) / gap
        closing_ms = closing_px * (Zv / f) * fps           # gap-closing speed in m/s
        ttc_boost = 0.0
        if closing_ms > 0.5:
            ttc_boost = float(np.clip((4.0 - d_real / closing_ms) / 4.0, 0, 1))
        r = prox * path * (0.75 + 0.25 * ttc_boost)
        if d_real < 3.5 and align > 0.6:                   # a car bearing down within ~3.5 m
            r = max(r, 0.72 + 0.25 * path)
        if r > best:
            best = r
            best_detail = {
                "driver_id": v.get("id"), "driver_cls": v.get("cls", "vehicle"),
                "d_real": d_real, "align": align, "prox": prox, "path": path,
                "ttc_boost": ttc_boost, "closing_ms": max(closing_ms, 0.0),
                "veh_speed": v_speed,
            }
    best = float(np.clip(best, 0, 1))
    if explain:
        return best, best_detail
    return best
