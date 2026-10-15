"""Near-miss event detection and characterisation.

A per-frame risk score is a signal; an *event* is what an operator acts on. This module
stitches the frame-by-frame interactions for each entity pair into contiguous episodes and
decides which are genuine near-misses, not merely "close", but a real conflict with
closing speed and a small realised miss distance, corroborated where possible by an
**evasive signature** (hard braking or a pedestrian's lateral jerk-away).

For every event it also computes the **early-warning time**: how many seconds before the
moment of closest approach the system first raised a HIGH alert. That lead time, "danger
flagged 2.7 s before the near-miss", is the metric that actually matters for prevention.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..config import RISK, FPS
from ..schema import RiskEvent, RiskBand, Recording, VehicleManeuver


def _zone_of(x: float, y: float, scene) -> str:
    """Name the region a point falls in (crosswalk, intersection core, or an approach)."""
    # crosswalks first
    from ..behavior.features import _point_in_poly
    names = ["S crosswalk", "N crosswalk", "W crosswalk", "E crosswalk"]
    for i, cw in enumerate(scene.crosswalks):
        if _point_in_poly(x, y, cw.polygon):
            return names[i] if i < len(names) else f"crosswalk {cw.id}"
    xmin, ymin, xmax, ymax = scene.ground_bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    if abs(x - cx) < 5 and abs(y - cy) < 5:
        return "intersection core"
    quad = ("N" if y > cy else "S") + ("E" if x > cx else "W")
    return f"{quad} approach"


class NearMissDetector:
    def __init__(self, min_gap_frames: int = 15):
        self.min_gap = min_gap_frames
        self._id = 0

    def detect(self, rec: Recording) -> list[RiskEvent]:
        scene = rec.scene
        # index entity states by (frame, id) for evasive-signature lookups
        estate = {(f.frame, e.id): e for f in rec.frames for e in f.entities}

        # gather per-pair interaction timelines
        pairs: dict[tuple, list] = defaultdict(list)
        for f in rec.frames:
            for it in f.interactions:
                pairs[(it.a_id, it.b_id)].append(it)

        events: list[RiskEvent] = []
        for (a_id, b_id), seq in pairs.items():
            seq.sort(key=lambda it: it.frame)
            for episode in self._split(seq):
                ev = self._build_event(a_id, b_id, episode, estate, scene)
                if ev is not None:
                    events.append(ev)
        events.sort(key=lambda e: (not e.is_near_miss, -e.peak_risk))
        return events

    def _split(self, seq):
        """Break a pair's timeline into episodes separated by long quiet gaps."""
        episodes, cur = [], []
        last = None
        for it in seq:
            if last is not None and it.frame - last > self.min_gap:
                episodes.append(cur); cur = []
            cur.append(it); last = it.frame
        if cur:
            episodes.append(cur)
        return episodes

    def _build_event(self, a_id, b_id, ep, estate, scene) -> RiskEvent | None:
        peak = max(ep, key=lambda it: it.risk_score)
        realized_sep = min(it.distance for it in ep)
        if peak.risk_score < RISK.band_high and realized_sep > RISK.nearmiss_dcpa * 2.5:
            return None  # not close in reality and never highly ranked -> not an event

        # The near-miss LABEL uses the *realised* closest separation actually observed
        # (min centre-to-centre distance), which is robust. Per-frame `it.dcpa` is a
        # *prediction* of future closest approach: great for early warning, but taking its
        # min over a long episode is noise-biased toward zero, so we do not label with it.
        realized_min = min(it.distance for it in ep)
        min_dcpa = realized_min
        min_pred_dcpa = min(it.dcpa for it in ep)
        finite_ttc = [it.ttc for it in ep if not math.isinf(it.ttc)]
        min_ttc = min(finite_ttc) if finite_ttc else math.inf
        max_closing = max(it.closing_speed for it in ep)
        # moment of closest approach = frame where they were actually closest
        coa = min(ep, key=lambda it: it.distance)

        # conflict eligibility: a near-miss requires the VRU to actually be in / entering
        # the roadway during the episode: a pedestrian safely on the sidewalk that a car
        # merely drives past is NOT a near-miss, however small the centre-to-centre gap.
        in_conflict = False
        for it in ep:
            ped = estate.get((it.frame, it.a_id))
            if ped is None:
                continue
            on_road = ped.attrs.get("on_road", 0.0) > 0.5
            dcurb = ped.attrs.get("dist_curb", 9.0)
            entering = (dcurb < 0.5) and ped.speed > 0.4 and it.features.get("path_conflict", 0) > 0.2
            if on_road or entering:
                in_conflict = True
                break

        is_near_miss = (in_conflict
                        and realized_min < RISK.nearmiss_dcpa
                        and max_closing >= RISK.nearmiss_min_closing_speed)

        # early warning: first HIGH alert -> the closest-approach frame
        high_frames = [it.frame for it in ep if it.risk_score >= RISK.band_high]
        first_high = min(high_frames) if high_frames else None
        early_warning = 0.0
        if first_high is not None and coa.frame >= first_high:
            early_warning = (coa.frame - first_high) / FPS

        band = self._band(peak.risk_score)
        evasive = self._evasive(a_id, b_id, ep, estate)
        loc = None
        est = estate.get((coa.frame, a_id))
        if est is not None:
            loc = [round(est.position.x, 2), round(est.position.y, 2)]
        zone = _zone_of(loc[0], loc[1], scene) if loc else None
        label = self._label(a_id, b_id, ep, estate, is_near_miss)

        ev = RiskEvent(
            id=self._id, a_id=a_id, b_id=b_id,
            start_frame=ep[0].frame, end_frame=ep[-1].frame,
            start_t=ep[0].t, end_t=ep[-1].t,
            peak_frame=peak.frame, peak_t=peak.t, peak_risk=peak.risk_score,
            min_ttc=min_ttc, min_dcpa=min_dcpa, max_closing_speed=max_closing,
            band=band, is_near_miss=is_near_miss, early_warning_s=round(early_warning, 2),
            location=loc, zone=zone, contributions=peak.contributions, evasive=evasive,
            label=label,
        )
        self._id += 1
        return ev

    def _evasive(self, a_id, b_id, ep, estate) -> dict:
        """Detect evasive signatures around the episode: hard vehicle braking / a
        pedestrian's sudden sideways (lateral) acceleration away from the conflict."""
        veh_decel = 0.0
        ped_lat = 0.0
        for it in ep:
            veh = estate.get((it.frame, b_id))
            ped = estate.get((it.frame, a_id))
            if veh is not None:
                hx, hy = math.cos(veh.heading), math.sin(veh.heading)
                al = veh.acceleration.x * hx + veh.acceleration.y * hy   # along-heading accel
                veh_decel = min(veh_decel, al)
                if veh.braking:
                    veh_decel = min(veh_decel, -RISK.evasive_decel)
            if ped is not None:
                # lateral acceleration = accel component perpendicular to the ped's heading
                # (m/s^2): a genuine swerve/jump-away, not filter jitter (jerk).
                nx, ny = -math.sin(ped.heading), math.cos(ped.heading)
                lat = abs(ped.acceleration.x * nx + ped.acceleration.y * ny)
                ped_lat = max(ped_lat, lat)
        return {
            "vehicle_hard_braking": bool(veh_decel <= -RISK.evasive_decel),
            "vehicle_min_decel": round(veh_decel, 2),
            "pedestrian_evasive": bool(ped_lat >= RISK.evasive_ped_lat_acc),
            "pedestrian_max_lat_acc": round(ped_lat, 2),
        }

    def _label(self, a_id, b_id, ep, estate, is_nm) -> str:
        veh = estate.get((ep[-1].frame, b_id)) or estate.get((ep[0].frame, b_id))
        turning = any(estate.get((it.frame, b_id)) and
                      estate[(it.frame, b_id)].maneuver in
                      (VehicleManeuver.TURNING_LEFT, VehicleManeuver.TURNING_RIGHT)
                      for it in ep)
        kind = "near miss" if is_nm else "conflict"
        if turning:
            return f"Turning-vehicle / pedestrian {kind}"
        onroad = any(estate.get((it.frame, a_id)) and estate[(it.frame, a_id)].attrs.get("on_road")
                     for it in ep)
        if onroad:
            return f"Mid-crossing {kind}"
        return f"Approach {kind}"

    @staticmethod
    def _band(risk: float) -> RiskBand:
        if risk >= RISK.band_critical:
            return RiskBand.CRITICAL
        if risk >= RISK.band_high:
            return RiskBand.HIGH
        if risk >= RISK.band_medium:
            return RiskBand.MEDIUM
        return RiskBand.LOW
