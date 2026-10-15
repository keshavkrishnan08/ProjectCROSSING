"""Behavioural feature extraction for pedestrians.

The crossing-intention and trajectory models consume a compact, physically-meaningful
feature vector *per frame*, assembled into a short temporal sequence. The same extractor
is used at training time (over whole simulated/annotated scenes) and at inference time
(over each pedestrian's rolling history), guaranteeing train/serve parity.

Features (per frame, all continuous & roughly normalised):

=====================  ==================================================================
name                   meaning
=====================  ==================================================================
speed                  ground speed (m/s)
accel                  signed acceleration along heading (m/s^2)
dist_curb              distance to the nearest road edge / curb (m; -ve once in the road)
on_road                1 if the point is within a road polygon
lat_progress           fraction of the way across the nearest crosswalk (0..1)
head_align             alignment of heading with the crossing axis (cos, -1..1)
gaze                   pose gaze cue toward traffic (-1..1; 0 if no pose)
ped_walk               pedestrian signal: +1 WALK, -1 DONT_WALK, 0 flashing/unknown
veh_dist               distance to nearest approaching vehicle (m, clipped)
veh_ttc                inverse TTC to nearest vehicle (1/s, clipped), bigger = more urgent
veh_speed              speed of nearest vehicle (m/s)
=====================  ==================================================================
"""
from __future__ import annotations

import math

import numpy as np

from ..schema import EntityKind, SignalState

FEATURE_NAMES = [
    "speed", "accel", "dist_curb", "on_road", "lat_progress", "head_align",
    "gaze", "ped_walk", "veh_dist", "veh_ttc", "veh_speed",
]
N_FEATURES = len(FEATURE_NAMES)


def _point_in_poly(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_poly_edges(x, y, poly) -> float:
    best = 1e18
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        t = ((x - ax) * abx + (y - ay) * aby) / (abx * abx + aby * aby + 1e-12)
        t = max(0.0, min(1.0, t))
        px, py = ax + t * abx, ay + t * aby
        best = min(best, math.hypot(x - px, y - py))
    return best


class SceneContext:
    """Pre-processes static scene geometry for fast per-frame feature queries."""

    def __init__(self, scene):
        self.scene = scene
        self.road_polys = []
        for r in scene.roads:
            self.road_polys.append(self._road_polygon(r))
        self.crosswalks = [np.array(c.polygon) for c in scene.crosswalks]
        self.cw_roads = [getattr(c, "road", None) for c in scene.crosswalks]
        self.lights = scene.traffic_lights

    @staticmethod
    def _road_polygon(road):
        pl = np.array(road.polyline)
        d = pl[-1] - pl[0]
        n = np.array([-d[1], d[0]]); n = n / (np.linalg.norm(n) + 1e-9)
        hw = road.width / 2.0
        a = pl[0] + n * hw; b = pl[-1] + n * hw
        c = pl[-1] - n * hw; e = pl[0] - n * hw
        return [a.tolist(), b.tolist(), c.tolist(), e.tolist()]

    def on_road(self, x, y) -> bool:
        return any(_point_in_poly(x, y, p) for p in self.road_polys)

    def dist_curb(self, x, y) -> float:
        """Signed distance to nearest road edge; negative inside a road."""
        d = min((_dist_point_poly_edges(x, y, p) for p in self.road_polys), default=1e9)
        return -d if self.on_road(x, y) else d

    def nearest_crosswalk(self, x, y):
        best, bd = None, 1e18
        for cw in self.crosswalks:
            c = cw.mean(axis=0)
            d = math.hypot(x - c[0], y - c[1])
            if d < bd:
                bd, best = d, cw
        return best, bd

    def ped_signal_num(self, frame: int, x, y) -> float:
        """Pedestrian signal aspect for the crosswalk nearest (x,y): +1/-1/0."""
        # find nearest pedestrian light
        best, bd, road = None, 1e18, None
        for lt in self.lights:
            if lt.controls != "pedestrian":
                continue
            d = math.hypot(x - lt.position[0], y - lt.position[1])
            if d < bd:
                bd, best = d, lt
        if best is None:
            return 0.0
        s = best.state_at(frame)
        if s == SignalState.WALK:
            return 1.0
        if s == SignalState.DONT_WALK:
            return -1.0
        return 0.0


def frame_features(ped_state, others, ctx: SceneContext, frame: int) -> np.ndarray:
    """Compute the per-frame feature vector for one pedestrian state."""
    x, y = ped_state.position.x, ped_state.position.y
    vx, vy = ped_state.velocity.x, ped_state.velocity.y
    speed = ped_state.speed
    # signed accel along heading
    hx, hy = math.cos(ped_state.heading), math.sin(ped_state.heading)
    accel = ped_state.acceleration.x * hx + ped_state.acceleration.y * hy

    dist_curb = ctx.dist_curb(x, y)
    on_road = 1.0 if dist_curb < 0 else 0.0

    cw, cwd = ctx.nearest_crosswalk(x, y)
    lat_progress, head_align = 0.0, 0.0
    if cw is not None:
        c = cw.mean(axis=0)
        # crossing axis: the crosswalk's long axis
        span = cw.max(axis=0) - cw.min(axis=0)
        axis = np.array([1.0, 0.0]) if span[0] >= span[1] else np.array([0.0, 1.0])
        # progress across (project onto axis, normalise by half-span)
        half = 0.5 * (span[0] if span[0] >= span[1] else span[1]) + 1e-6
        lat_progress = float(np.clip(0.5 + ((x - c[0]) * axis[0] + (y - c[1]) * axis[1]) / (2 * half), 0, 1))
        if speed > 0.2:
            head_align = float(abs(hx * axis[0] + hy * axis[1]))

    gaze = float(ped_state.attrs.get("gaze_cue", 0.0))
    ped_walk = ctx.ped_signal_num(frame, x, y)

    # nearest approaching vehicle
    veh_dist, veh_ttc, veh_speed = 30.0, 0.0, 0.0
    bd = 1e18
    for o in others:
        if o.kind not in (EntityKind.VEHICLE, EntityKind.CYCLIST):
            continue
        dx, dy = o.position.x - x, o.position.y - y
        d = math.hypot(dx, dy)
        if d < bd:
            bd = d
            rvx, rvy = o.velocity.x - vx, o.velocity.y - vy
            closing = -(dx * rvx + dy * rvy) / (d + 1e-6)   # +ve = approaching
            veh_dist = min(d, 30.0)
            veh_ttc = float(np.clip(closing / (d + 1e-6), 0.0, 3.0)) if closing > 0 else 0.0
            veh_speed = o.speed
    return np.array([
        speed / 3.0,
        np.clip(accel / 3.0, -3, 3),
        np.clip(dist_curb / 5.0, -3, 3),
        on_road,
        lat_progress,
        head_align,
        gaze,
        ped_walk,
        veh_dist / 30.0,
        veh_ttc,
        veh_speed / 12.0,
    ], dtype=float)
