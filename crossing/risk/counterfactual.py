"""Counterfactual simulation, decision support, not just description.

Given a detected event, ask: *what if the vehicle had been 5 mph slower?* or *what if the
pedestrian had waited one second?* CROSSING answers by replaying the two actors' **actual
recorded trajectories** under a modified control and recomputing the realised miss distance
and risk. Slowing the vehicle rescales how far along its own path it has travelled at each
instant; delaying the pedestrian time-shifts their real path. Nothing is invented, we
reuse the trajectories the scene actually produced.

This is explicitly a *model simulation*: it estimates how the conflict geometry would have
changed, not a proof that the outcome was caused by any single factor.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import MPH_PER_MS, RISK
from ..schema import Recording, RiskEvent
from ..interaction import metrics as M
from .engine import PhysicsRisk


def _entity_track(rec: Recording, eid: int, f0: int, f1: int):
    """(times, positions[N,2]) for entity ``eid`` across frames [f0, f1]."""
    ts, ps = [], []
    for f in rec.frames:
        if f.frame < f0 or f.frame > f1:
            continue
        for e in f.entities:
            if e.id == eid:
                ts.append(f.t); ps.append([e.position.x, e.position.y])
                break
    return np.array(ts), np.array(ps)


def _arclen(ps: np.ndarray):
    d = np.linalg.norm(np.diff(ps, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _pos_at_arclen(ps: np.ndarray, cum: np.ndarray, s: float):
    s = float(np.clip(s, 0, cum[-1]))
    i = int(np.searchsorted(cum, s)) - 1
    i = max(0, min(i, len(ps) - 2))
    seg = cum[i + 1] - cum[i]
    frac = 0.0 if seg < 1e-9 else (s - cum[i]) / seg
    return ps[i] + frac * (ps[i + 1] - ps[i])


def _min_geometry(veh_pos, ped_pos, times, radius):
    """Minimum separation and the geometry/risk at that instant, given aligned samples."""
    seps = np.linalg.norm(veh_pos - ped_pos, axis=1)
    k = int(np.argmin(seps))
    dmin = float(seps[k])
    # instantaneous velocities by finite difference
    def vel(arr, i):
        i = min(max(i, 1), len(arr) - 1)
        dt = max(times[i] - times[i - 1], 1e-3)
        return (arr[i] - arr[i - 1]) / dt
    vv, pv = vel(veh_pos, k), vel(ped_pos, k)
    closing = M.closing_speed(ped_pos[k], pv, veh_pos[k], vv)
    ttc = M.ttc(ped_pos[k], pv, veh_pos[k], vv, radius)
    conflict = M.path_conflict(ped_pos[k], pv, veh_pos[k], vv, radius=radius)
    feats = {
        "inv_ttc": 0.0 if math.isinf(ttc) else 1.0 / max(ttc, 0.15),
        "dcpa": dmin, "closing_speed": max(0.0, closing), "path_conflict": conflict,
        "veh_speed": float(np.linalg.norm(vv)),
    }
    # physics risk here is geometry-driven (no hardcoded on-road boost) so it *responds*
    # to the counterfactual: the caller anchors the baseline to the event's real risk.
    risk, _ = PhysicsRisk().score(feats)
    return dmin, (None if math.isinf(ttc) else ttc), risk


class CounterfactualEngine:
    def __init__(self, radius: float = 2.0):
        self.radius = radius

    def _window(self, ev: RiskEvent):
        return ev.start_frame - 20, ev.end_frame + 40

    def vehicle_speed_sweep(self, rec: Recording, ev: RiskEvent,
                            factors=(1.0, 0.9, 0.8, 0.7, 0.6)) -> list[dict]:
        f0, f1 = self._window(ev)
        tv, pv = _entity_track(rec, ev.b_id, f0, f1)
        tp, pp = _entity_track(rec, ev.a_id, f0, f1)
        if len(pv) < 3 or len(pp) < 3:
            return []
        # common time base = pedestrian frames; interpolate vehicle arclength timing
        cum = _arclen(pv)
        # original vehicle arclength as a function of time (assume its own timeline tv)
        s_of_t = np.interp(tp, tv, cum)
        base_speed = np.mean(np.linalg.norm(np.diff(pv, axis=0), axis=1) /
                             np.maximum(np.diff(tv), 1e-3)) if len(tv) > 1 else 0.0
        rows = []
        for k in factors:
            veh_pos = np.array([_pos_at_arclen(pv, cum, k * s) for s in s_of_t])
            dmin, ttc, risk = _min_geometry(veh_pos, pp, tp, self.radius)
            rows.append((k, dmin, ttc, risk))
        return self._finish(rows, ev, base_speed)

    # counterfactual outcome severity falls off smoothly as the *realised* miss distance
    # grows, anchored so the unmodified config reproduces the event's reported risk. This is
    # the interpretable, honest thing to show: "the more they miss by, the safer it was".
    _SAFE_DIST = 3.2   # metres of separation at which the encounter is essentially benign

    def _finish(self, rows, ev, base_speed, is_wait=False):
        base_dmin = rows[0][1] if rows else 0.0
        denom = max(self._SAFE_DIST - min(base_dmin, self._SAFE_DIST - 0.5), 0.5)
        out = []
        for row in rows:
            k, dmin, ttc, _phys = row
            frac = float(np.clip((self._SAFE_DIST - dmin) / denom, 0.0, 1.0))
            cf_risk = round(ev.peak_risk * frac, 0)
            if is_wait:
                d = {"delay_s": k}
            else:
                d = {"factor": k, "delta_mph": round((1 - k) * base_speed * MPH_PER_MS, 1),
                     "veh_speed_mph": round(k * base_speed * MPH_PER_MS, 1)}
            d.update({
                "min_dcpa": round(dmin, 2),
                "ttc": None if ttc is None else round(ttc, 2),
                "risk": cf_risk,
                "near_miss": bool(dmin < RISK.nearmiss_dcpa),
            })
            out.append(d)
        return out

    def pedestrian_wait_sweep(self, rec: Recording, ev: RiskEvent,
                              delays=(0.0, 0.5, 1.0, 1.5, 2.0)) -> list[dict]:
        f0, f1 = self._window(ev)
        tv, pv = _entity_track(rec, ev.b_id, f0, f1)
        tp, pp = _entity_track(rec, ev.a_id, f0, f1)
        if len(pv) < 3 or len(pp) < 3:
            return []
        # align vehicle onto pedestrian timeline
        veh_on_tp = np.column_stack([np.interp(tp, tv, pv[:, 0]), np.interp(tp, tv, pv[:, 1])])
        rows = []
        for d in delays:
            # delayed pedestrian = where it was `d` seconds earlier (holds start position before t0)
            ped_delayed = np.column_stack([np.interp(tp - d, tp, pp[:, 0]),
                                           np.interp(tp - d, tp, pp[:, 1])])
            dmin, ttc, risk = _min_geometry(veh_on_tp, ped_delayed, tp, self.radius)
            rows.append((d, dmin, ttc, risk))
        return self._finish(rows, ev, base_speed=0.0, is_wait=True)

    def for_event(self, rec: Recording, ev: RiskEvent) -> dict:
        return {
            "event_id": ev.id,
            "vehicle_speed": self.vehicle_speed_sweep(rec, ev),
            "pedestrian_wait": self.pedestrian_wait_sweep(rec, ev),
            "note": "Model simulation over recorded trajectories; not causal proof.",
        }
