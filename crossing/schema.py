"""The unified data model for CROSSING.

A street is *not* a bag of bounding boxes, it is a network of entities whose states
and relationships evolve over time. This module is the single vocabulary every stage
speaks: perception fills :class:`Detection`, tracking promotes them to :class:`Track`,
the world model resolves :class:`Person` / :class:`Vehicle` states in ground metres,
the interaction engine emits :class:`Interaction`, and the risk engine emits
:class:`RiskEvent` / :class:`NearMiss`. A whole clip is a :class:`Recording`.

Everything is a plain dataclass with ``to_dict`` / ``from_dict`` so a scene round-trips
losslessly through JSON (the web demo and the event DB both consume that JSON).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ======================================================================================
# Enumerations: the discrete state vocabularies
# ======================================================================================
class EntityKind(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    CYCLIST = "cyclist"
    INFRASTRUCTURE = "infrastructure"


class Behavior(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"
    CROSSING = "crossing"
    WAITING = "waiting"
    TURNING = "turning"
    UNKNOWN = "unknown"


class Attention(str, Enum):
    LOOKING = "looking_at_traffic"
    NOT_LOOKING = "not_looking"
    UNCERTAIN = "uncertain"


class VehicleManeuver(str, Enum):
    CRUISE = "cruise"
    ACCELERATING = "accelerating"
    BRAKING = "braking"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    STOPPED = "stopped"


class SignalState(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    WALK = "walk"
    DONT_WALK = "dont_walk"
    FLASHING = "flashing"
    UNKNOWN = "unknown"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ======================================================================================
# Small geometry helpers (kept dependency-free so schema imports cleanly everywhere)
# ======================================================================================
@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    @property
    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    @staticmethod
    def from_list(v) -> "Vec2":
        return Vec2(float(v[0]), float(v[1]))


@dataclass
class BBox:
    """Image-space axis-aligned box, pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre, the ground-contact point projected by the homography."""
        return (self.cx, self.y2)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @staticmethod
    def from_list(v) -> "BBox":
        return BBox(float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


# ======================================================================================
# Perception + tracking level
# ======================================================================================
@dataclass
class Detection:
    """A single detector output for one frame (image space)."""

    frame: int
    cls: str
    bbox: BBox
    score: float
    keypoints: Optional[list[list[float]]] = None  # [[x,y,conf], ...] pose, optional
    # ground-truth back-reference (only present for simulated / annotated data; enables eval)
    gt_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "cls": self.cls,
            "bbox": self.bbox.to_list(),
            "score": self.score,
            "keypoints": self.keypoints,
            "gt_id": self.gt_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(
            frame=d["frame"],
            cls=d["cls"],
            bbox=BBox.from_list(d["bbox"]),
            score=d["score"],
            keypoints=d.get("keypoints"),
            gt_id=d.get("gt_id"),
        )


# ======================================================================================
# World-model entity states (ground space)
# ======================================================================================
@dataclass
class EntityState:
    """One entity at one instant, resolved onto the ground plane.

    Shared base for pedestrians, vehicles and cyclists so the interaction engine can
    treat them uniformly; kind-specific attributes live in the ``attrs`` bag.
    """

    id: int
    kind: EntityKind
    cls: str
    frame: int
    t: float                       # seconds
    # image space (for overlay rendering)
    bbox: Optional[BBox] = None
    # ground space (metres): the physics frame
    position: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    acceleration: Vec2 = field(default_factory=Vec2)
    heading: float = 0.0           # radians
    speed: float = 0.0             # m/s (|velocity|)
    jerk: float = 0.0              # m/s^3 magnitude
    # confidence / observability
    score: float = 1.0
    visibility: float = 1.0        # 1 = fully visible, 0 = fully occluded
    age: int = 0                   # frames tracked
    # behavioural / semantic layer
    behavior: Behavior = Behavior.UNKNOWN
    attention: Attention = Attention.UNCERTAIN
    maneuver: VehicleManeuver = VehicleManeuver.CRUISE
    crossing_intention: float = 0.0        # P(cross) in [0,1]
    intention_std: float = 0.0             # predictive uncertainty
    braking: bool = False
    risk_score: float = 0.0                # aggregate over this entity's interactions
    # relationships (interaction ids this entity participates in this frame)
    interaction_ids: list[int] = field(default_factory=list)
    # freeform (lane, turn_state, pose_summary, etc.)
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "cls": self.cls,
            "frame": self.frame,
            "t": round(self.t, 4),
            "bbox": self.bbox.to_list() if self.bbox else None,
            "position": [round(self.position.x, 3), round(self.position.y, 3)],
            "velocity": [round(self.velocity.x, 3), round(self.velocity.y, 3)],
            "acceleration": [round(self.acceleration.x, 3), round(self.acceleration.y, 3)],
            "heading": round(self.heading, 4),
            "speed": round(self.speed, 3),
            "jerk": round(self.jerk, 3),
            "score": round(self.score, 3),
            "visibility": round(self.visibility, 3),
            "age": self.age,
            "behavior": self.behavior.value,
            "attention": self.attention.value,
            "maneuver": self.maneuver.value,
            "crossing_intention": round(self.crossing_intention, 4),
            "intention_std": round(self.intention_std, 4),
            "braking": self.braking,
            "risk_score": round(self.risk_score, 2),
            "interaction_ids": self.interaction_ids,
            "attrs": self.attrs,
        }

    @staticmethod
    def from_dict(d: dict) -> "EntityState":
        return EntityState(
            id=d["id"],
            kind=EntityKind(d["kind"]),
            cls=d["cls"],
            frame=d["frame"],
            t=d["t"],
            bbox=BBox.from_list(d["bbox"]) if d.get("bbox") else None,
            position=Vec2.from_list(d["position"]),
            velocity=Vec2.from_list(d["velocity"]),
            acceleration=Vec2.from_list(d["acceleration"]),
            heading=d["heading"],
            speed=d["speed"],
            jerk=d.get("jerk", 0.0),
            score=d.get("score", 1.0),
            visibility=d.get("visibility", 1.0),
            age=d.get("age", 0),
            behavior=Behavior(d.get("behavior", "unknown")),
            attention=Attention(d.get("attention", "uncertain")),
            maneuver=VehicleManeuver(d.get("maneuver", "cruise")),
            crossing_intention=d.get("crossing_intention", 0.0),
            intention_std=d.get("intention_std", 0.0),
            braking=d.get("braking", False),
            risk_score=d.get("risk_score", 0.0),
            interaction_ids=d.get("interaction_ids", []),
            attrs=d.get("attrs", {}),
        )


# ======================================================================================
# Predicted futures
# ======================================================================================
@dataclass
class TrajectoryMode:
    """One hypothesised future for an entity, with a probability and waypoints."""

    prob: float
    label: str                       # e.g. "continues crossing", "stops", "turns back"
    points: list[list[float]]        # [[t, x, y], ...] in ground metres
    std: Optional[list[float]] = None  # per-step positional sigma (m), optional

    def to_dict(self) -> dict:
        return {"prob": round(self.prob, 4), "label": self.label,
                "points": [[round(p[0], 3), round(p[1], 3), round(p[2], 3)] for p in self.points],
                "std": self.std}

    @staticmethod
    def from_dict(d: dict) -> "TrajectoryMode":
        return TrajectoryMode(d["prob"], d["label"], d["points"], d.get("std"))


@dataclass
class Prediction:
    """The multi-modal predicted future for a single entity at a frame."""

    entity_id: int
    frame: int
    modes: list[TrajectoryMode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "frame": self.frame,
                "modes": [m.to_dict() for m in self.modes]}

    @staticmethod
    def from_dict(d: dict) -> "Prediction":
        return Prediction(d["entity_id"], d["frame"],
                          [TrajectoryMode.from_dict(m) for m in d["modes"]])


# ======================================================================================
# Interaction + risk level
# ======================================================================================
@dataclass
class Interaction:
    """A time-resolved relationship between two entities (usually person <-> vehicle)."""

    id: int
    frame: int
    t: float
    a_id: int                       # typically the vulnerable road user
    b_id: int                       # typically the vehicle
    distance: float                 # current separation (m)
    closing_speed: float            # rate of approach along the line of sight (m/s, +=closing)
    rel_speed: float                # |relative velocity| (m/s)
    ttc: float                      # time to collision (s), inf if diverging
    dcpa: float                     # distance at closest point of approach (m)
    tcpa: float                     # time to closest point of approach (s)
    pet: float                      # post-encroachment time (s), inf if paths don't cross
    path_conflict: float            # 0-1 overlap of predicted paths
    risk_score: float = 0.0         # 0-100
    band: RiskBand = RiskBand.LOW
    features: dict = field(default_factory=dict)   # engineered features fed to the risk model
    contributions: dict = field(default_factory=dict)  # risk decomposition {factor: points}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = self.band.value
        for k in ("distance", "closing_speed", "rel_speed", "dcpa", "tcpa", "risk_score"):
            d[k] = round(d[k], 3)
        d["ttc"] = None if math.isinf(self.ttc) else round(self.ttc, 3)
        d["pet"] = None if math.isinf(self.pet) else round(self.pet, 3)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Interaction":
        return Interaction(
            id=d["id"], frame=d["frame"], t=d["t"], a_id=d["a_id"], b_id=d["b_id"],
            distance=d["distance"], closing_speed=d["closing_speed"], rel_speed=d["rel_speed"],
            ttc=math.inf if d["ttc"] is None else d["ttc"],
            dcpa=d["dcpa"], tcpa=d["tcpa"],
            pet=math.inf if d.get("pet") is None else d["pet"],
            path_conflict=d.get("path_conflict", 0.0),
            risk_score=d.get("risk_score", 0.0),
            band=RiskBand(d.get("band", "low")),
            features=d.get("features", {}),
            contributions=d.get("contributions", {}),
        )


@dataclass
class RiskEvent:
    """A sustained high-risk episode between two entities (aggregated over frames)."""

    id: int
    a_id: int
    b_id: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    peak_frame: int
    peak_t: float
    peak_risk: float
    min_ttc: float
    min_dcpa: float
    max_closing_speed: float
    band: RiskBand
    is_near_miss: bool
    early_warning_s: float          # lead time between first HIGH alert and peak danger
    location: Optional[list[float]] = None   # ground [x,y] of peak
    zone: Optional[str] = None               # named region / intersection
    contributions: dict = field(default_factory=dict)
    evasive: dict = field(default_factory=dict)   # detected evasive signatures
    label: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = self.band.value
        d["min_ttc"] = None if math.isinf(self.min_ttc) else round(self.min_ttc, 3)
        d["min_dcpa"] = round(self.min_dcpa, 3)
        d["peak_risk"] = round(self.peak_risk, 2)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RiskEvent":
        return RiskEvent(
            id=d["id"], a_id=d["a_id"], b_id=d["b_id"],
            start_frame=d["start_frame"], end_frame=d["end_frame"],
            start_t=d["start_t"], end_t=d["end_t"],
            peak_frame=d["peak_frame"], peak_t=d["peak_t"], peak_risk=d["peak_risk"],
            min_ttc=math.inf if d["min_ttc"] is None else d["min_ttc"],
            min_dcpa=d["min_dcpa"], max_closing_speed=d["max_closing_speed"],
            band=RiskBand(d["band"]), is_near_miss=d["is_near_miss"],
            early_warning_s=d.get("early_warning_s", 0.0),
            location=d.get("location"), zone=d.get("zone"),
            contributions=d.get("contributions", {}), evasive=d.get("evasive", {}),
            label=d.get("label", ""),
        )


# ======================================================================================
# Infrastructure (static context): changes what a given kinematic state *means*
# ======================================================================================
@dataclass
class Crosswalk:
    id: int
    polygon: list[list[float]]      # ground-space vertices (m)
    signal_id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrafficLightState:
    frame: int
    state: SignalState

    def to_dict(self) -> dict:
        return {"frame": self.frame, "state": self.state.value}


@dataclass
class TrafficLight:
    id: int
    position: list[float]           # ground [x,y]
    controls: str = "vehicle"       # "vehicle" | "pedestrian"
    states: list[TrafficLightState] = field(default_factory=list)

    def state_at(self, frame: int) -> SignalState:
        s = SignalState.UNKNOWN
        for st in self.states:
            if st.frame <= frame:
                s = st.state
            else:
                break
        return s

    def to_dict(self) -> dict:
        return {"id": self.id, "position": self.position, "controls": self.controls,
                "states": [s.to_dict() for s in self.states]}


@dataclass
class Road:
    id: int
    polyline: list[list[float]]     # centreline (m)
    lanes: int = 2
    direction: float = 0.0          # heading of travel (rad)
    speed_limit_mph: float = 30.0
    width: float = 7.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    """Static scene description shared by every frame of a recording."""

    name: str
    homography: list[list[float]]   # 3x3 image->ground
    image_w: int
    image_h: int
    ground_bounds: list[float]      # [xmin, ymin, xmax, ymax] metres
    roads: list[Road] = field(default_factory=list)
    crosswalks: list[Crosswalk] = field(default_factory=list)
    traffic_lights: list[TrafficLight] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "homography": self.homography,
            "image_w": self.image_w, "image_h": self.image_h,
            "ground_bounds": self.ground_bounds,
            "roads": [r.to_dict() for r in self.roads],
            "crosswalks": [c.to_dict() for c in self.crosswalks],
            "traffic_lights": [t.to_dict() for t in self.traffic_lights],
        }

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return Scene(
            name=d["name"], homography=d["homography"],
            image_w=d["image_w"], image_h=d["image_h"],
            ground_bounds=d["ground_bounds"],
            roads=[Road(**r) for r in d.get("roads", [])],
            crosswalks=[Crosswalk(**c) for c in d.get("crosswalks", [])],
            traffic_lights=[
                TrafficLight(
                    id=t["id"], position=t["position"], controls=t.get("controls", "vehicle"),
                    states=[TrafficLightState(s["frame"], SignalState(s["state"]))
                            for s in t.get("states", [])],
                )
                for t in d.get("traffic_lights", [])
            ],
        )


# ======================================================================================
# Per-frame world state and the full recording
# ======================================================================================
@dataclass
class WorldState:
    """The complete inferred state of the street at one instant."""

    frame: int
    t: float
    entities: list[EntityState] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    signals: dict = field(default_factory=dict)   # {light_id: state} snapshot

    def to_dict(self) -> dict:
        return {
            "frame": self.frame, "t": round(self.t, 4),
            "entities": [e.to_dict() for e in self.entities],
            "interactions": [i.to_dict() for i in self.interactions],
            "predictions": [p.to_dict() for p in self.predictions],
            "signals": self.signals,
        }

    @staticmethod
    def from_dict(d: dict) -> "WorldState":
        return WorldState(
            frame=d["frame"], t=d["t"],
            entities=[EntityState.from_dict(e) for e in d["entities"]],
            interactions=[Interaction.from_dict(i) for i in d.get("interactions", [])],
            predictions=[Prediction.from_dict(p) for p in d.get("predictions", [])],
            signals=d.get("signals", {}),
        )


@dataclass
class Recording:
    """A processed clip: static scene + timeline of world states + detected events."""

    scene: Scene
    fps: float
    frames: list[WorldState] = field(default_factory=list)
    events: list[RiskEvent] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "fps": self.fps,
            "frames": [f.to_dict() for f in self.frames],
            "events": [e.to_dict() for e in self.events],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Recording":
        return Recording(
            scene=Scene.from_dict(d["scene"]),
            fps=d["fps"],
            frames=[WorldState.from_dict(f) for f in d["frames"]],
            events=[RiskEvent.from_dict(e) for e in d.get("events", [])],
            meta=d.get("meta", {}),
        )
