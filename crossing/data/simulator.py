"""A physically-grounded traffic micro-simulator, CROSSING's synthetic data engine.

Real footage (PIE/JAAD) plugs in through :mod:`crossing.data.pie_adapter`, but to develop,
test, and *evaluate* the whole intelligence stack today we need scenes with known ground
truth: exact trajectories, behaviours, crossing intentions, and, crucially, genuine
vehicle-pedestrian conflicts. This module generates them.

What is modelled (all in ground metres, right-hand traffic like Toronto/PIE):

* a signalised 4-way intersection with four crosswalks and a phase-cycling controller;
* vehicles with **IDM** car-following, stop-line compliance, permissive left/right turns
  along smooth arcs, and probabilistic yielding to pedestrians;
* pedestrians with an approach->wait->gap-acceptance->cross state machine, a latent
  crossing **intention** that ramps observably, gaze/attention, and social-force spacing;
* seeded *conflict scenarios* (permissive-left-vs-pedestrian, red-light jaywalk,
  occluded-emergence) so near-misses genuinely arise from the kinematics.

The simulator then **renders** each actor to an image-space detection through the scene
homography, adds calibrated noise, misses, false positives, and depth-ordered occlusion,
and back-references ground truth by ``gt_id`` so every downstream stage can be measured.
Every scene is fully determined by its integer ``seed`` -> independent scenes give
leakage-free train/val/test splits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import IMAGE_W, IMAGE_H, FPS, MPH_PER_MS
from ..schema import (
    BBox, Detection, Scene, Road, Crosswalk, TrafficLight, TrafficLightState,
    SignalState, Behavior, Attention, VehicleManeuver, EntityKind,
)
from ..worldmodel.homography import synthetic_camera_homography, Homography

# --------------------------------------------------------------------------------------
# Fixed intersection geometry (ground metres)
# --------------------------------------------------------------------------------------
CX, CY = 20.0, 23.0          # intersection centre
HALF = 3.5                    # half road width (2 lanes each direction)
LANE = 1.75                   # lane centre offset
GROUND_BOUNDS = (-2.0, 0.0, 42.0, 46.0)

# lane centres
EB_Y = CY - LANE             # east-bound vehicles (moving +x), south half
WB_Y = CY + LANE             # west-bound (moving -x), north half
NB_X = CX + LANE             # north-bound (moving +y), east half
SB_X = CX - LANE             # south-bound (moving -y), west half


@dataclass
class SimConfig:
    seed: int = 7
    duration_s: float = 60.0
    fps: float = FPS
    vehicle_rate: float = 0.22        # spawns/sec/approach (Poisson-ish)
    ped_rate: float = 0.16            # spawns/sec/corner
    turn_left_prob: float = 0.22
    turn_right_prob: float = 0.20
    yield_prob: float = 0.93          # P(a vehicle yields to a ped in its path)
    jaywalk_prob: float = 0.18        # P(a ped will accept a risky gap / cross on DONT_WALK)
    distracted_prob: float = 0.30     # P(a ped not looking at traffic while crossing)
    n_scripted_conflicts: int = 3     # guaranteed high-quality conflict episodes
    detector_noise_px: float = 4.0
    detector_dropout: float = 0.05
    detector_fp_rate: float = 0.012
    occlusion_drop: float = 0.55      # visibility below which a detection is dropped


# --------------------------------------------------------------------------------------
# Signal controller
# --------------------------------------------------------------------------------------
class SignalController:
    """Two-phase signal with yellow/all-red; exposes vehicle + pedestrian aspects.

    Phase A: EW vehicles GREEN, NS RED, N/S crosswalks WALK (peds cross the stopped NS road,
             concurrent with, and thus conflicting with, permissive EW turns).
    Phase B: the mirror image.
    """

    def __init__(self, green=14.0, yellow=3.0, allred=2.0, offset=0.0):
        self.green, self.yellow, self.allred = green, yellow, allred
        self.cycle = 2 * (green + yellow + allred)
        self.offset = offset

    def _phase(self, t: float):
        p = (t + self.offset) % self.cycle
        g, y, r = self.green, self.yellow, self.allred
        if p < g:
            return "A", "green"
        if p < g + y:
            return "A", "yellow"
        if p < g + y + r:
            return "A", "allred"
        p -= g + y + r
        if p < g:
            return "B", "green"
        if p < g + y:
            return "B", "yellow"
        return "B", "allred"

    def vehicle_signal(self, t: float, axis: str) -> SignalState:
        """axis in {'EW','NS'}."""
        phase, sub = self._phase(t)
        active = "A" if axis == "EW" else "B"
        if phase == active:
            return {"green": SignalState.GREEN, "yellow": SignalState.YELLOW,
                    "allred": SignalState.RED}[sub]
        return SignalState.RED

    def ped_signal(self, t: float, road: str) -> SignalState:
        """road = the road the crosswalk spans ('EW' or 'NS').

        Pedestrians crossing a road walk WHILE that road's vehicles are stopped, i.e. a
        crosswalk over the NS road shows WALK during Phase A (EW green / NS red).
        """
        phase, sub = self._phase(t)
        walk_phase = "A" if road == "NS" else "B"
        if phase == walk_phase and sub == "green":
            return SignalState.WALK
        if phase == walk_phase and sub == "yellow":
            return SignalState.FLASHING
        return SignalState.DONT_WALK


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
@dataclass
class Vehicle:
    id: int
    cls: str
    approach: str                 # 'E','W','N','S' (entry side)
    axis: str                     # 'EW' or 'NS'
    route: str                    # 'through','left','right'
    path: np.ndarray              # (K,2) ground waypoints
    s: float = 0.0                # arclength progress along path
    v: float = 8.0                # speed m/s
    a: float = 0.0
    length: float = 4.6
    width: float = 1.9
    desired_v: float = 9.0
    aggressive: bool = False
    alive: bool = True
    braking: bool = False
    prev_v: float = 8.0
    no_yield: bool = False          # scripted conflict vehicles ignore ped yielding
    ignore_signal: bool = False     # scripted red-runners
    no_lead: bool = False           # scripted focal vehicles ignore car-following
    seg_len: np.ndarray = field(default=None, repr=False)
    total_len: float = 0.0

    def setup(self):
        d = np.diff(self.path, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.total_len = float(self.seg_len.sum())

    def pose(self):
        """Return (x, y, heading) at current arclength s."""
        return _path_pose(self.path, self.seg_len, self.s)

    def stop_line_s(self) -> float:
        """Arclength at which this vehicle reaches the intersection stop line."""
        # first waypoint entering the central box
        for i, p in enumerate(self.path):
            if abs(p[0] - CX) <= HALF + 0.5 and abs(p[1] - CY) <= HALF + 0.5:
                return float(self.seg_len[:i].sum()) if i > 0 else 0.0
        return self.total_len


@dataclass
class Pedestrian:
    id: int
    corner: str                   # 'SW','SE','NW','NE'
    crosswalk: str                # 'S','N','W','E' (which crosswalk band)
    road: str                     # road it spans: 'NS' or 'EW'
    start: np.ndarray             # ground start
    goal: np.ndarray              # ground goal (opposite curb)
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    desired_speed: float = 1.4
    state: str = "approach"       # approach|wait|cross|done
    intention: float = 0.0        # observable P(cross), ramps up
    committed: bool = False
    jaywalker: bool = False
    distracted: bool = False
    looking: bool = True
    curb: np.ndarray = None       # curb entry point
    alive: bool = True
    wait_t: float = 0.0
    evasive: float = 0.0          # lateral evasive accel magnitude (for near-miss signature)


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------
def _path_pose(path: np.ndarray, seg_len: np.ndarray, s: float):
    if s <= 0:
        h = math.atan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
        return path[0, 0], path[0, 1], h
    acc = 0.0
    for i in range(len(seg_len)):
        if acc + seg_len[i] >= s:
            frac = (s - acc) / max(seg_len[i], 1e-6)
            p = path[i] + frac * (path[i + 1] - path[i])
            h = math.atan2(path[i + 1, 1] - path[i, 1], path[i + 1, 0] - path[i, 0])
            return float(p[0]), float(p[1]), h
        acc += seg_len[i]
    h = math.atan2(path[-1, 1] - path[-2, 1], path[-1, 0] - path[-2, 0])
    return float(path[-1, 0]), float(path[-1, 1]), h


def _arc(center, radius, a0, a1, n=14):
    ang = np.linspace(a0, a1, n)
    return np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)])


# Per-approach lane geometry: entry lane (into the box) and where a through vehicle exits.
# Approach = entry side. Right-hand traffic:
#   W -> east-bound (lane EB_Y, +x) | E -> west-bound (WB_Y, -x)
#   N -> south-bound (SB_X, -y)     | S -> north-bound (NB_X, +y)
_ENTRY_BOX = {                      # first box-edge point on the entry lane
    "W": (CX - HALF, EB_Y), "E": (CX + HALF, WB_Y),
    "N": (SB_X, CY + HALF), "S": (NB_X, CY - HALF),
}
_EXIT_BOX = {                       # box-edge point for each destination lane
    "eastbound": (CX + HALF, EB_Y), "westbound": (CX - HALF, WB_Y),
    "northbound": (NB_X, CY + HALF), "southbound": (SB_X, CY - HALF),
}
_EXIT_FAR = {
    "eastbound": (CX + 26, EB_Y), "westbound": (CX - 26, WB_Y),
    "northbound": (NB_X, CY + 26), "southbound": (SB_X, CY - 26),
}
# Correct destination for each (approach, turn) under right-hand rules.
_TURN_DEST = {
    ("W", "left"): "northbound", ("W", "right"): "southbound",
    ("E", "left"): "southbound", ("E", "right"): "northbound",
    ("N", "left"): "eastbound", ("N", "right"): "westbound",
    ("S", "left"): "westbound", ("S", "right"): "eastbound",
}


def _vehicle_path(approach: str, route: str) -> tuple[np.ndarray, str]:
    """Build a ground waypoint path for a vehicle entering from ``approach``."""
    far = 26.0
    axis = "EW" if approach in ("E", "W") else "NS"
    entry_far = {
        "W": [CX - far, EB_Y], "E": [CX + far, WB_Y],
        "N": [SB_X, CY + far], "S": [NB_X, CY - far],
    }[approach]
    entry = np.array([entry_far, list(_ENTRY_BOX[approach])])
    if route == "through":
        dest = {"W": "eastbound", "E": "westbound", "N": "southbound", "S": "northbound"}[approach]
        path = np.vstack([entry, [list(_EXIT_BOX[dest]), list(_EXIT_FAR[dest])]])
    else:
        dest = _TURN_DEST[(approach, route)]
        arc = _turn_arc(approach, dest)
        path = np.vstack([entry, arc, [list(_EXIT_FAR[dest])]])
    return path.astype(float), axis


def _turn_arc(approach: str, dest: str) -> np.ndarray:
    """Smooth quadratic-Bezier quarter-turn from the entry box-edge to the exit box-edge.

    The control point is the corner where the entry lane's line meets the exit lane's line,
    which yields a tight radius for right turns and a wide sweep for lefts, naturally.
    """
    p0 = np.array(_ENTRY_BOX[approach], float)
    p1 = np.array(_EXIT_BOX[dest], float)
    # entry line is along the entry lane (fixed lateral coord); exit line along exit lane.
    # corner = (exit-lane's fixed axis, entry-lane's fixed axis)
    if approach in ("W", "E"):        # entry lane fixes y, exit lane fixes x
        ctrl = np.array([p1[0], p0[1]], float)
    else:                              # entry lane fixes x, exit lane fixes y
        ctrl = np.array([p0[0], p1[1]], float)
    ts = np.linspace(0, 1, 12)[:, None]
    return (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * ctrl + ts ** 2 * p1


# corner / crosswalk geometry for pedestrians
CORNERS = {
    "SW": np.array([CX - HALF - 2.5, CY - HALF - 2.5]),
    "SE": np.array([CX + HALF + 2.5, CY - HALF - 2.5]),
    "NW": np.array([CX - HALF - 2.5, CY + HALF + 2.5]),
    "NE": np.array([CX + HALF + 2.5, CY + HALF + 2.5]),
}


def _crosswalk_geom():
    """Return crosswalk polygons + the road each spans + signal axis."""
    b = HALF + 0.4
    e = HALF + 2.6
    cws = {
        "S": dict(road="NS", poly=[[CX - b, CY - e], [CX + b, CY - e], [CX + b, CY - b], [CX - b, CY - b]]),
        "N": dict(road="NS", poly=[[CX - b, CY + b], [CX + b, CY + b], [CX + b, CY + e], [CX - b, CY + e]]),
        "W": dict(road="EW", poly=[[CX - e, CY - b], [CX - b, CY - b], [CX - b, CY + b], [CX - e, CY + b]]),
        "E": dict(road="EW", poly=[[CX + b, CY - b], [CX + e, CY - b], [CX + e, CY + b], [CX + b, CY + b]]),
    }
    return cws


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------
@dataclass
class SimResult:
    scene: Scene
    fps: float
    n_frames: int
    detections: list          # per frame: list[Detection]
    gt_frames: list           # per frame: list[dict] ground-truth entity states
    gt_conflicts: list        # scripted conflict metadata (for risk-model eval)
    homography: Homography


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------
class TrafficSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.dt = 1.0 / cfg.fps
        self.H = synthetic_camera_homography(IMAGE_W, IMAGE_H, GROUND_BOUNDS)
        self.signals = SignalController(offset=float(self.rng.uniform(0, 8)))
        self.cws = _crosswalk_geom()
        self._veh_id = 0
        self._ped_id = 10_000
        self.vehicles: list[Vehicle] = []
        self.peds: list[Pedestrian] = []
        self.gt_conflicts: list = []

    # -- scene ------------------------------------------------------------------------
    def build_scene(self) -> Scene:
        roads = [
            Road(id=0, polyline=[[CX - 26, CY], [CX + 26, CY]], lanes=2, direction=0.0,
                 speed_limit_mph=30.0, width=2 * HALF),
            Road(id=1, polyline=[[CX, CY - 26], [CX, CY + 26]], lanes=2, direction=math.pi / 2,
                 speed_limit_mph=30.0, width=2 * HALF),
        ]
        crosswalks = [Crosswalk(id=i, polygon=c["poly"], signal_id=i)
                      for i, (k, c) in enumerate(self.cws.items())]
        # traffic lights (vehicle aspects for EW and NS) + generate their state timelines
        n = int(self.cfg.duration_s * self.cfg.fps)
        tl = []
        for lid, (axis, pos) in enumerate([("EW", [CX + HALF + 1, CY + HALF + 1]),
                                           ("NS", [CX + HALF + 1, CY - HALF - 1])]):
            states, last = [], None
            for f in range(n):
                s = self.signals.vehicle_signal(f * self.dt, axis)
                if s != last:
                    states.append(TrafficLightState(f, s))
                    last = s
            tl.append(TrafficLight(id=lid, position=pos, controls="vehicle", states=states))
        # pedestrian signals as lights too (ids 2,3)
        for lid, (road, pos) in enumerate([("NS", [CX - HALF - 1, CY - HALF - 1]),
                                           ("EW", [CX - HALF - 1, CY + HALF + 1])], start=2):
            states, last = [], None
            for f in range(n):
                s = self.signals.ped_signal(f * self.dt, road)
                if s != last:
                    states.append(TrafficLightState(f, s))
                    last = s
            tl.append(TrafficLight(id=lid, position=pos, controls="pedestrian", states=states))
        return Scene(
            name=f"intersection_seed{self.cfg.seed}",
            homography=self.H.to_list(), image_w=IMAGE_W, image_h=IMAGE_H,
            ground_bounds=list(GROUND_BOUNDS), roads=roads, crosswalks=crosswalks,
            traffic_lights=tl,
        )

    # -- spawning ---------------------------------------------------------------------
    def _spawn_vehicle(self):
        approach = self.rng.choice(["E", "W", "N", "S"])
        r = self.rng.random()
        route = "left" if r < self.cfg.turn_left_prob else (
            "right" if r < self.cfg.turn_left_prob + self.cfg.turn_right_prob else "through")
        path, axis = _vehicle_path(approach, route)
        cls = self.rng.choice(["car", "car", "car", "truck", "bus"], p=[0.55, 0.2, 0.13, 0.08, 0.04])
        dims = {"car": (4.6, 1.9), "truck": (7.5, 2.4), "bus": (11.0, 2.5)}[cls]
        v = Vehicle(id=self._veh_id, cls=cls, approach=approach, axis=axis, route=route,
                    path=path, v=float(self.rng.uniform(7.5, 10.5)),
                    desired_v=float(self.rng.uniform(8.0, 11.5)),
                    length=dims[0], width=dims[1],
                    aggressive=bool(self.rng.random() < 0.25))
        v.setup()
        # avoid spawning on top of a same-approach vehicle
        for o in self.vehicles:
            if o.alive and o.approach == approach and o.s < 6.0:
                return
        self._veh_id += 1
        self.vehicles.append(v)

    def _spawn_ped(self, forced=None):
        if forced is None:
            corner = self.rng.choice(list(CORNERS.keys()))
            # pick a crosswalk adjacent to this corner
            options = {"SW": ["S", "W"], "SE": ["S", "E"], "NW": ["N", "W"], "NE": ["N", "E"]}[corner]
            cwk = self.rng.choice(options)
        else:
            corner, cwk = forced
        road = self.cws[cwk]["road"]
        poly = np.array(self.cws[cwk]["poly"])
        # start at the near curb, goal at far curb of this crosswalk
        if road == "NS":  # crosses NS road -> walks in x
            y0 = poly[:, 1].mean()
            if "W" in corner:
                start = np.array([CX - HALF - 2.0, y0]); goal = np.array([CX + HALF + 2.0, y0])
            else:
                start = np.array([CX + HALF + 2.0, y0]); goal = np.array([CX - HALF - 2.0, y0])
        else:             # crosses EW road -> walks in y
            x0 = poly[:, 0].mean()
            if "S" in corner:
                start = np.array([x0, CY - HALF - 2.0]); goal = np.array([x0, CY + HALF + 2.0])
            else:
                start = np.array([x0, CY + HALF + 2.0]); goal = np.array([x0, CY - HALF - 2.0])
        curb = start + 0.85 * (goal - start) / np.linalg.norm(goal - start) * 0.0  # placeholder
        # curb point = just at the edge of the road
        direction = (goal - start) / np.linalg.norm(goal - start)
        curb = start + direction * 1.6
        p = Pedestrian(
            id=self._ped_id, corner=corner, crosswalk=cwk, road=road,
            start=start.astype(float), goal=goal.astype(float), pos=start.astype(float).copy(),
            desired_speed=float(self.rng.uniform(1.2, 1.7)),
            jaywalker=bool(self.rng.random() < self.cfg.jaywalk_prob),
            distracted=bool(self.rng.random() < self.cfg.distracted_prob),
            curb=curb.astype(float),
        )
        self._ped_id += 1
        self.peds.append(p)
        return p

    # -- vehicle dynamics (IDM + signal + yield) --------------------------------------
    def _vehicle_accel(self, v: Vehicle, t: float) -> float:
        a_max, b, T, s0 = 1.8, 2.6, 1.4, 2.5
        if v.aggressive:
            a_max, b, T, s0 = 2.4, 3.2, 0.9, 1.8
        dv_free = a_max * (1 - (v.v / max(v.desired_v, 0.1)) ** 4)

        # gap to lead vehicle on same approach & route ahead
        s_gap, dv_lead = 1e9, 0.0
        if not v.no_lead:
            for o in self.vehicles:
                if o is v or not o.alive or o.approach != v.approach:
                    continue
                if o.s > v.s and (o.s - v.s) < s_gap:
                    s_gap = o.s - v.s
                    dv_lead = v.v - o.v

        # stop line if red/yellow and not yet committed into box
        sig = self.signals.vehicle_signal(t, v.axis)
        sl = v.stop_line_s()
        if v.ignore_signal:
            sig = SignalState.GREEN
        if sig in (SignalState.RED, SignalState.YELLOW) and v.s < sl - 0.5:
            gap_stop = sl - v.s
            # don't slam if yellow and already too close (dilemma zone) -> proceed if very close
            if not (sig == SignalState.YELLOW and gap_stop < 4.0):
                if gap_stop < s_gap:
                    s_gap = gap_stop
                    dv_lead = v.v

        # yield to pedestrians in the vehicle's near path (sample the curved path ahead)
        px, py, _ = v.pose()
        if not v.no_yield:
            lookahead = np.linspace(0.0, 12.0, 9)
            for p in self.peds:
                if not p.alive or p.state not in ("cross", "wait"):
                    continue
                d_ped = np.hypot(p.pos[0] - px, p.pos[1] - py)
                if d_ped > 12.0:
                    continue
                # minimum distance from the ped to the vehicle's forward path, and its
                # arclength (so the vehicle stops SHORT of the crossing point)
                best_d, best_s = 1e9, 0.0
                for ds in lookahead:
                    fx, fy, _ = _path_pose(v.path, v.seg_len, min(v.total_len, v.s + ds))
                    dd = math.hypot(p.pos[0] - fx, p.pos[1] - fy)
                    if dd < best_d:
                        best_d, best_s = dd, ds
                # ped must be within a car-width corridor and ahead of the vehicle
                if best_d < 2.6 and best_s > 0.3:
                    yields = self.rng.random() < (0.85 if v.aggressive else self.cfg.yield_prob)
                    if yields:
                        gap_ped = max(0.3, best_s - 4.0)   # stop ~4 m short of the conflict
                        if gap_ped < s_gap:
                            s_gap = gap_ped
                            dv_lead = v.v

        s_star = s0 + max(0.0, v.v * T + v.v * dv_lead / (2 * math.sqrt(a_max * b)))
        dv_int = -a_max * (s_star / max(s_gap, 0.5)) ** 2
        a = dv_free + dv_int
        return float(np.clip(a, -8.0, 2.6))

    # -- pedestrian dynamics ----------------------------------------------------------
    def _step_ped(self, p: Pedestrian, t: float):
        direction = (p.goal - p.pos)
        dist_goal = np.linalg.norm(direction)
        if dist_goal < 0.6:
            p.state = "done"; p.alive = False; return
        direction = direction / max(dist_goal, 1e-6)

        ped_sig = self.signals.ped_signal(t, p.road)
        at_curb = np.linalg.norm(p.pos - p.curb) < 0.7 or (p.state in ("wait", "cross"))
        in_road = abs(p.pos[0] - CX) < HALF + 0.6 if p.road == "NS" else abs(p.pos[1] - CY) < HALF + 0.6

        # --- intention dynamics (observable ramp) ---
        approach_frac = 1.0 - np.clip(np.linalg.norm(p.pos - p.curb) / 4.0, 0, 1)
        target_int = 0.15 + 0.55 * approach_frac
        if ped_sig == SignalState.WALK:
            target_int = max(target_int, 0.85)
        if p.jaywalker and self._safe_gap(p) > 0.0:
            target_int = max(target_int, 0.8)
        if p.committed:
            target_int = 0.97
        if p.state == "done":
            target_int = 0.0
        p.intention += (target_int - p.intention) * min(1.0, 3.0 * self.dt)

        # --- state machine ---
        if p.state == "approach":
            step = direction * p.desired_speed
            if np.linalg.norm(p.pos - p.curb) < 0.7:
                p.state = "wait"
        elif p.state == "wait":
            step = np.zeros(2)
            p.wait_t += self.dt
            can_go = (ped_sig == SignalState.WALK)
            if p.jaywalker and self._safe_gap(p) > 1.5:
                can_go = True
            if (ped_sig == SignalState.FLASHING) and self.rng.random() < 0.02:
                can_go = True  # rush the flashing hand
            if can_go and p.wait_t > 0.2:
                p.committed = True; p.state = "cross"
        elif p.state == "cross":
            step = direction * p.desired_speed
            # evasive reaction to an imminent vehicle (creates near-miss signature)
            hazard, hv = self._nearest_hazard(p)
            if hazard is not None:
                ttc = self._ped_vehicle_ttc(p, hazard)
                if ttc < 1.3:
                    # jump forward or freeze
                    if self.rng.random() < 0.6:
                        step = direction * (p.desired_speed * 1.9)  # sprint through
                    else:
                        step = step * 0.1                            # freeze
                    lateral = np.array([-direction[1], direction[0]])
                    evade = lateral * float(self.rng.uniform(-1.0, 1.0)) * 2.0
                    step = step + evade
                    p.evasive = float(np.linalg.norm(evade))
                else:
                    p.evasive = 0.0
        else:
            step = np.zeros(2)

        # social force separation
        for q in self.peds:
            if q is p or not q.alive:
                continue
            d = p.pos - q.pos
            dn = np.linalg.norm(d)
            if 1e-3 < dn < 1.2:
                step = step + (d / dn) * (1.2 - dn) * 0.9

        # attention / gaze
        oncoming = self._oncoming_dir(p)
        p.looking = (not p.distracted) and (self.rng.random() > 0.15)
        p.vel = step
        p.pos = p.pos + step * self.dt

    def _oncoming_dir(self, p):
        # unit vector toward oncoming traffic lane on the road being crossed
        if p.road == "NS":
            return np.array([1.0, 0.0]) if p.goal[0] > p.pos[0] else np.array([-1.0, 0.0])
        return np.array([0.0, 1.0]) if p.goal[1] > p.pos[1] else np.array([0.0, -1.0])

    def _nearest_hazard(self, p):
        best, bestd = None, 1e9
        for v in self.vehicles:
            if not v.alive:
                continue
            px, py, _ = v.pose()
            d = math.hypot(px - p.pos[0], py - p.pos[1])
            if d < bestd and d < 14.0:
                bestd, best = d, v
        return best, bestd

    def _ped_vehicle_ttc(self, p, v):
        px, py, _ = v.pose()
        vv = self._vehicle_velocity(v)
        rel_p = p.pos - np.array([px, py])
        rel_v = p.vel - vv
        # closing time along line of approach
        vv_dot = rel_v @ rel_v
        if vv_dot < 1e-6:
            return 1e9
        tstar = -(rel_p @ rel_v) / vv_dot
        if tstar < 0:
            return 1e9
        closest = rel_p + rel_v * tstar
        if np.linalg.norm(closest) > 3.0:
            return 1e9
        return tstar

    def _safe_gap(self, p):
        """A crude gap-acceptance measure (seconds of clearance) for jaywalkers."""
        v, d = self._nearest_hazard(p)
        if v is None:
            return 5.0
        speed = max(v.v, 0.1)
        return d / speed - 1.0

    def _vehicle_velocity(self, v: Vehicle) -> np.ndarray:
        x0, y0, h = v.pose()
        return np.array([math.cos(h), math.sin(h)]) * v.v

    # -- scripted conflicts -----------------------------------------------------------
    def _seed_conflicts(self):
        """Return a list of (trigger_time, kind, setup_fn) guaranteed conflict episodes."""
        cfg = self.cfg
        times = np.linspace(6, cfg.duration_s - 8, cfg.n_scripted_conflicts)
        base = ["permissive_left", "red_jaywalk", "occluded_emerge"]
        kinds = [base[i % 3] for i in range(cfg.n_scripted_conflicts)]
        self.rng.shuffle(kinds)
        self._scripted = list(zip(times.tolist(), kinds))
        self._scripted_done = [False] * len(self._scripted)

    def _synchronize(self, v: Vehicle, p: Pedestrian, lead_m: float):
        """Place a vehicle and pedestrian so they arrive at their true path crossing
        simultaneously, guaranteeing a tight approach.

        The conflict point is where the vehicle's *lane* path crosses the pedestrian's
        crossing segment (start->goal), not the crosswalk midline, which the vehicle
        never actually drives over."""
        cpt, s_star = _seg_path_closest(v.path, v.seg_len, p.start, p.goal)
        v.s = max(0.4, s_star - lead_m)
        t_arrive = lead_m / max(v.v, 0.1)
        pdir = (p.goal - p.start)
        pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
        ped_pos = cpt - pdir * (p.desired_speed * t_arrive)
        # keep the ped between its start curb and the conflict point
        along = float(np.dot(ped_pos - p.start, pdir))
        along = min(max(along, 0.0), float(np.linalg.norm(cpt - p.start)))
        p.pos = (p.start + pdir * along).astype(float)
        p.committed = True; p.state = "cross"

    def _fire_conflict(self, idx, kind, t):
        self._scripted_done[idx] = True
        left_conflict_cw = {"W": "N", "E": "S", "N": "E", "S": "W"}
        if kind == "permissive_left":
            approach = self.rng.choice(["W", "E", "N", "S"])
            path, axis = _vehicle_path(approach, "left")
            v = Vehicle(id=self._veh_id, cls="car", approach=approach, axis=axis, route="left",
                        path=path, v=8.5, desired_v=10.0, aggressive=True, no_yield=True,
                        no_lead=True)
            v.setup(); self._veh_id += 1; self.vehicles.append(v)
            cwk = left_conflict_cw[approach]
            corner = self._corner_for_cw(cwk, near="left")
            p = self._spawn_ped(forced=(corner, cwk))
            p.distracted = True; p.desired_speed = 1.1
            self._synchronize(v, p, lead_m=11.0)
            self.gt_conflicts.append(dict(t=t, kind=kind, veh=v.id, ped=p.id, cw=cwk))
        elif kind == "red_jaywalk":
            approach = self.rng.choice(["W", "E", "N", "S"])
            path, axis = _vehicle_path(approach, "through")
            v = Vehicle(id=self._veh_id, cls="car", approach=approach, axis=axis, route="through",
                        path=path, v=11.0, desired_v=11.5, aggressive=True, no_yield=True,
                        ignore_signal=True, no_lead=True)
            v.setup(); self._veh_id += 1; self.vehicles.append(v)
            cwk = {"W": "W", "E": "E", "N": "N", "S": "S"}[approach]
            corner = self._corner_for_cw(cwk, near="left")
            p = self._spawn_ped(forced=(corner, cwk))
            p.jaywalker = True; p.desired_speed = 1.3
            self._synchronize(v, p, lead_m=15.0)
            self.gt_conflicts.append(dict(t=t, kind=kind, veh=v.id, ped=p.id, cw=cwk))
        else:  # occluded_emerge: distracted ped bolts into a fast non-yielding car's path,
               # with a large vehicle stopped nearby that intermittently occludes detection
            approach = self.rng.choice(["W", "E"])
            pathf, axisf = _vehicle_path(approach, "through")
            fast = Vehicle(id=self._veh_id, cls="car", approach=approach, axis=axisf, route="through",
                           path=pathf, v=10.5, desired_v=11.0, aggressive=True, no_yield=True,
                           ignore_signal=True, no_lead=True)
            fast.setup(); self._veh_id += 1; self.vehicles.append(fast)
            cwk = {"W": "W", "E": "E"}[approach]
            corner = self._corner_for_cw(cwk, near="left")
            p = self._spawn_ped(forced=(corner, cwk))
            p.jaywalker = True; p.distracted = True; p.desired_speed = 1.4
            self._synchronize(fast, p, lead_m=16.0)
            # a bus stopped in the oncoming lane near the crosswalk (large occluder in-frame)
            opp = {"W": "E", "E": "W"}[approach]
            pathb, axisb = _vehicle_path(opp, "through")
            big = Vehicle(id=self._veh_id, cls="bus", approach=opp, axis=axisb, route="through",
                          path=pathb, v=0.6, desired_v=1.0, length=11.0, width=2.5,
                          no_lead=True, ignore_signal=True)
            big.setup()
            cpt_b, s_b = _seg_path_closest(big.path, big.seg_len, p.start, p.goal)
            big.s = max(0.4, s_b - 1.0)
            self._veh_id += 1; self.vehicles.append(big)
            self.gt_conflicts.append(dict(t=t, kind=kind, veh=fast.id, ped=p.id, cw=cwk, occluder=big.id))

    @staticmethod
    def _corner_for_cw(cwk: str, near: str = "left") -> str:
        """A corner adjacent to crosswalk ``cwk`` to start the pedestrian from."""
        opts = {"S": ("SW", "SE"), "N": ("NW", "NE"), "W": ("SW", "NW"), "E": ("SE", "NE")}[cwk]
        return opts[0] if near == "left" else opts[1]

    # -- rendering to detections ------------------------------------------------------
    def _ground_bbox(self, gx, gy, length, width, height_m) -> BBox:
        fx, fy = self.H.point_ground_to_image(gx, gy)
        fx2, fy2 = self.H.point_ground_to_image(gx, gy + 0.6)  # 0.6 m farther (up-image)
        ppm = abs(fy - fy2) / 0.6                              # local vertical px per metre
        px_h = height_m * ppm * 1.35
        px_w = max(width, length * 0.5) * ppm * 1.05
        return BBox(fx - px_w / 2, fy - px_h, fx + px_w / 2, fy)

    def _in_frame(self, gx, gy) -> bool:
        """Is a ground point within the camera's field of view (padded ground bounds)?"""
        xmin, ymin, xmax, ymax = GROUND_BOUNDS
        return (xmin - 0.3) <= gx <= (xmax + 0.3) and (ymin + 0.3) <= gy <= (ymax + 0.3)

    def _render(self, frame, actors):
        """actors: list of dict(gt_id, cls, gx, gy, length, width, height, kind)."""
        cfg = self.cfg
        # a fixed camera only sees its FOV: skip actors outside the visible ground region
        actors = [a for a in actors if self._in_frame(a["gx"], a["gy"])]
        # depth order: nearer to camera = larger image-y (bottom); occluders are nearer
        boxes = []
        for a in actors:
            bb = self._ground_bbox(a["gx"], a["gy"], a["length"], a["width"], a["height"])
            boxes.append((a, bb))
        boxes.sort(key=lambda ab: ab[1].y2)  # far (top) -> near (bottom)
        dets = []
        for i, (a, bb) in enumerate(boxes):
            # occlusion by any nearer box
            vis = 1.0
            for j in range(i + 1, len(boxes)):
                _, bb2 = boxes[j]
                inter = bb.iou(bb2) * (bb.w * bb.h + bb2.w * bb2.h)  # rough overlap area proxy
                ox = max(0.0, min(bb.x2, bb2.x2) - max(bb.x1, bb2.x1))
                oy = max(0.0, min(bb.y2, bb2.y2) - max(bb.y1, bb2.y1))
                cover = (ox * oy) / max(bb.w * bb.h, 1e-6)
                vis -= 0.9 * cover
            vis = float(np.clip(vis, 0.0, 1.0))
            a["_vis"] = vis
            # drop fully off-image
            if bb.x2 < 0 or bb.x1 > IMAGE_W or bb.y2 < 0 or bb.y1 > IMAGE_H:
                continue
            # occlusion / random dropout
            if vis < cfg.occlusion_drop and self.rng.random() < 0.85:
                continue
            if self.rng.random() < cfg.detector_dropout:
                continue
            n = cfg.detector_noise_px
            nb = BBox(bb.x1 + self.rng.normal(0, n), bb.y1 + self.rng.normal(0, n),
                      bb.x2 + self.rng.normal(0, n), bb.y2 + self.rng.normal(0, n))
            score = float(np.clip(0.9 * vis + self.rng.normal(0, 0.05) + 0.1, 0.2, 0.99))
            kp = self._pose_keypoints(a, nb) if a["kind"] == "person" else None
            dets.append(Detection(frame=frame, cls=a["cls"], bbox=nb, score=score,
                                  keypoints=kp, gt_id=a["gt_id"]))
        # false positives
        n_fp = self.rng.poisson(cfg.detector_fp_rate)
        for _ in range(n_fp):
            fx = float(self.rng.uniform(0, IMAGE_W)); fy = float(self.rng.uniform(IMAGE_H * 0.3, IMAGE_H))
            w = float(self.rng.uniform(30, 90)); h = float(self.rng.uniform(60, 160))
            dets.append(Detection(frame=frame, cls="person", bbox=BBox(fx, fy - h, fx + w, fy),
                                  score=float(self.rng.uniform(0.2, 0.4)), gt_id=None))
        return dets

    def _pose_keypoints(self, a, bb: BBox):
        """A crude 5-point pose (head, shoulders, hips) encoding gaze/attention."""
        cx = bb.cx
        head_y = bb.y1 + 0.12 * bb.h
        # gaze offset: looking toward oncoming shifts head x
        gaze = a.get("_gaze", 0.0)
        return [
            [cx + gaze * 0.15 * bb.w, head_y, 0.9],                 # head/nose
            [cx - 0.18 * bb.w, bb.y1 + 0.3 * bb.h, 0.8],            # L shoulder
            [cx + 0.18 * bb.w, bb.y1 + 0.3 * bb.h, 0.8],            # R shoulder
            [cx - 0.12 * bb.w, bb.y1 + 0.55 * bb.h, 0.7],          # L hip
            [cx + 0.12 * bb.w, bb.y1 + 0.55 * bb.h, 0.7],          # R hip
        ]

    # -- main loop --------------------------------------------------------------------
    def run(self) -> SimResult:
        cfg = self.cfg
        scene = self.build_scene()
        n = int(cfg.duration_s * cfg.fps)
        self._seed_conflicts()
        detections, gt_frames = [], []

        for f in range(n):
            t = f * self.dt
            # spawns (Poisson thinning)
            if self.rng.random() < cfg.vehicle_rate * self.dt * 4:
                self._spawn_vehicle()
            if self.rng.random() < cfg.ped_rate * self.dt * 4:
                self._spawn_ped()
            for i, (ct, kind) in enumerate(self._scripted):
                if (not self._scripted_done[i]) and t >= ct:
                    self._fire_conflict(i, kind, t)

            # step vehicles
            for v in self.vehicles:
                if not v.alive:
                    continue
                a = self._vehicle_accel(v, t)
                v.prev_v = v.v
                v.v = max(0.0, v.v + a * self.dt)
                v.a = a
                v.braking = a < -1.5
                v.s += v.v * self.dt
                if v.s >= v.total_len - 0.2:
                    v.alive = False
            # step peds
            for p in self.peds:
                if p.alive:
                    self._step_ped(p, t)

            # assemble GT + render
            actors, gt = [], []
            for v in self.vehicles:
                if not v.alive:
                    continue
                x, y, h = v.pose()
                vv = self._vehicle_velocity(v)
                man = VehicleManeuver.CRUISE
                if v.route == "left" and abs(v.s - v.stop_line_s()) < 6:
                    man = VehicleManeuver.TURNING_LEFT
                elif v.route == "right" and abs(v.s - v.stop_line_s()) < 6:
                    man = VehicleManeuver.TURNING_RIGHT
                if v.braking:
                    man = VehicleManeuver.BRAKING
                elif v.v < 0.5:
                    man = VehicleManeuver.STOPPED
                height = {"car": 1.5, "truck": 3.2, "bus": 3.3}[v.cls]
                actors.append(dict(gt_id=v.id, cls=v.cls, gx=x, gy=y, length=v.length,
                                   width=v.width, height=height, kind="vehicle"))
                gt.append(dict(id=v.id, kind="vehicle", cls=v.cls, x=x, y=y,
                               vx=float(vv[0]), vy=float(vv[1]), heading=h, speed=v.v,
                               behavior="unknown",
                               maneuver=man.value, braking=v.braking, intention=0.0,
                               attention="uncertain", vis=1.0, route=v.route))
            for p in self.peds:
                if not p.alive:
                    continue
                gaze = 1.0 if p.looking else 0.0
                actors.append(dict(gt_id=p.id, cls="person", gx=p.pos[0], gy=p.pos[1],
                                   length=0.6, width=0.6, height=1.72, kind="person", _gaze=gaze))
                beh = {"approach": Behavior.WALKING, "wait": Behavior.WAITING,
                       "cross": Behavior.CROSSING, "done": Behavior.STANDING}[p.state]
                spd = float(np.linalg.norm(p.vel))
                if p.state == "cross" and spd > 2.3:
                    beh = Behavior.RUNNING
                att = Attention.LOOKING if p.looking else Attention.NOT_LOOKING
                gt.append(dict(id=p.id, kind="person", cls="person", x=float(p.pos[0]),
                               y=float(p.pos[1]), vx=float(p.vel[0]), vy=float(p.vel[1]),
                               heading=math.atan2(p.vel[1], p.vel[0]) if spd > 1e-3 else 0.0,
                               speed=spd, behavior=beh.value, maneuver="cruise",
                               braking=False, intention=float(p.intention),
                               attention=att.value, vis=1.0, evasive=float(p.evasive)))
            dets = self._render(f, actors)
            # propagate rendered visibility back into GT + mark field-of-view membership
            vis_map = {a["gt_id"]: a.get("_vis", 1.0) for a in actors}
            for g in gt:
                g["in_frame"] = self._in_frame(g["x"], g["y"])
                g["vis"] = float(vis_map.get(g["id"], 1.0 if g["in_frame"] else 0.0))
            detections.append(dets)
            gt_frames.append(gt)

        return SimResult(scene=scene, fps=cfg.fps, n_frames=n, detections=detections,
                         gt_frames=gt_frames, gt_conflicts=self.gt_conflicts, homography=self.H)


def _point_seg_dist(p, a, b):
    p = np.asarray(p, float); a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _seg_path_closest(path: np.ndarray, seg_len: np.ndarray, a, b):
    """Closest approach between a polyline ``path`` and a segment ``a``->``b``.

    Returns ``(point_on_path, arclength_s)``. Used to find the true vehicle<->pedestrian
    crossing point (vehicle lane path vs. pedestrian crossing line)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    ab2 = max(float(ab @ ab), 1e-9)
    best = (path[0].copy(), 0.0, 1e18)
    acc = 0.0
    for i in range(len(seg_len)):
        p0, p1 = path[i], path[i + 1]
        for tt in np.linspace(0, 1, 9):
            q = p0 + tt * (p1 - p0)
            u = np.clip(float((q - a) @ ab) / ab2, 0.0, 1.0)
            proj = a + u * ab
            d = float(np.linalg.norm(q - proj))
            if d < best[2]:
                best = (q.copy(), acc + tt * seg_len[i], d)
        acc += seg_len[i]
    return best[0], float(best[1])


def _closest_s_to_point(path: np.ndarray, seg_len: np.ndarray, point) -> float:
    """Arclength along ``path`` at the point of closest approach to ``point``."""
    point = np.asarray(point, float)
    best_s, best_d, acc = 0.0, 1e18, 0.0
    for i in range(len(seg_len)):
        a, b = path[i], path[i + 1]
        ab = b - a
        denom = max(float(ab @ ab), 1e-9)
        tt = np.clip(float((point - a) @ ab) / denom, 0.0, 1.0)
        proj = a + tt * ab
        d = float(np.linalg.norm(point - proj))
        if d < best_d:
            best_d = d
            best_s = acc + tt * seg_len[i]
        acc += seg_len[i]
    return float(best_s)


def simulate(seed: int = 7, duration_s: float = 60.0, **kw) -> SimResult:
    """Convenience entry point."""
    cfg = SimConfig(seed=seed, duration_s=duration_s, **kw)
    return TrafficSimulator(cfg).run()
