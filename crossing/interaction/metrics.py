"""Conflict-geometry primitives — the physics of a near-miss.

A single number like TTC is not enough: a person and a car 5 m apart can have a tiny TTC
yet be perfectly safe if they are moving apart. CROSSING therefore computes a *family* of
complementary surrogate-safety measures, all in ground metres/seconds, from the two
actors' positions, velocities and footprints:

* **closing_speed** — rate at which the gap is shrinking (m/s; +ve = approaching).
* **TTC** — time until their footprints touch on the current headings (∞ if they miss).
* **DCPA / TCPA** — distance and time at the closest point of approach (the miss distance).
* **PET** — post-encroachment time: the gap between one leaving the crossing point and the
  other arriving (∞ if their straight-line paths do not cross).
* **path_conflict** — 0..1 overlap of the two predicted corridors.

Every function is closed-form and unit-tested (``tests/test_metrics.py``).
"""
from __future__ import annotations

import math

import numpy as np

INF = float("inf")


def _v2(v):
    return np.array([v.x, v.y], float) if hasattr(v, "x") else np.asarray(v, float)


def relative(pa, va, pb, vb):
    pa, va, pb, vb = map(_v2, (pa, va, pb, vb))
    return pa - pb, va - vb


def distance(pa, pb) -> float:
    pa, pb = _v2(pa), _v2(pb)
    return float(np.linalg.norm(pa - pb))


def closing_speed(pa, va, pb, vb) -> float:
    """Rate of decrease of separation (m/s). Positive = closing."""
    p, v = relative(pa, va, pb, vb)
    d = np.linalg.norm(p)
    if d < 1e-6:
        return 0.0
    return float(-(p @ v) / d)


def cpa(pa, va, pb, vb) -> tuple[float, float]:
    """Closest point of approach: returns (DCPA metres, TCPA seconds)."""
    p, v = relative(pa, va, pb, vb)
    vv = float(v @ v)
    if vv < 1e-9:
        return float(np.linalg.norm(p)), 0.0
    t = -float(p @ v) / vv
    if t < 0:                       # already diverging; closest point is now
        return float(np.linalg.norm(p)), 0.0
    closest = p + v * t
    return float(np.linalg.norm(closest)), float(t)


def ttc(pa, va, pb, vb, radius: float) -> float:
    """Time until the two footprints (combined ``radius``) first touch, on current
    velocities. ∞ if they never come within ``radius``."""
    p, v = relative(pa, va, pb, vb)
    a = float(v @ v)
    b = 2.0 * float(p @ v)
    c = float(p @ p) - radius * radius
    if c <= 0:
        return 0.0                  # already overlapping
    if a < 1e-9:
        return INF                  # no relative motion
    disc = b * b - 4 * a * c
    if disc < 0:
        return INF                  # trajectories miss
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    ts = [t for t in (t1, t2) if t >= 0]
    return min(ts) if ts else INF


def pet(pa, va, pb, vb, horizon: float = 6.0) -> float:
    """Post-encroachment time: |Δ arrival| at the straight-line path crossing point.

    ∞ if the two rays do not cross within ``horizon`` seconds of travel.
    """
    pa, va, pb, vb = map(_v2, (pa, va, pb, vb))
    # solve pa + va*s = pb + vb*u  ->  [va, -vb][s,u]^T = pb - pa
    A = np.column_stack([va, -vb])
    rhs = pb - pa
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return INF
    su = np.linalg.solve(A, rhs)
    s, u = float(su[0]), float(su[1])
    if s < 0 or u < 0 or s > horizon or u > horizon:
        return INF
    return abs(s - u)


def path_conflict(pa, va, pb, vb, horizon: float = 4.0, radius: float = 2.0) -> float:
    """0..1 overlap of the two predicted straight corridors over ``horizon`` seconds.

    Sampled minimum separation mapped through the combined footprint radius: separation
    <= radius -> 1, >= 3*radius -> 0.
    """
    pa, va, pb, vb = map(_v2, (pa, va, pb, vb))
    ts = np.linspace(0, horizon, 17)
    seps = np.linalg.norm((pa + np.outer(ts, va)) - (pb + np.outer(ts, vb)), axis=1)
    dmin = float(seps.min())
    return float(np.clip((3 * radius - dmin) / (2 * radius), 0.0, 1.0))


def all_metrics(pa, va, pb, vb, radius: float) -> dict:
    d = distance(pa, pb)
    dcpa, tcpa = cpa(pa, va, pb, vb)
    return {
        "distance": d,
        "closing_speed": closing_speed(pa, va, pb, vb),
        "rel_speed": float(np.linalg.norm(relative(pa, va, pb, vb)[1])),
        "ttc": ttc(pa, va, pb, vb, radius),
        "dcpa": dcpa,
        "tcpa": tcpa,
        "pet": pet(pa, va, pb, vb),
        "path_conflict": path_conflict(pa, va, pb, vb, radius=radius),
    }
