"""World-model assembly: tracks -> ground-referenced :class:`EntityState`s per frame.

The world model is the boundary between *pixels* and *physics*. For each confirmed track
it projects the bounding-box foot point through the scene homography onto the road plane,
maintains a :class:`GroundTrack` (Kalman kinematics), classifies the entity kind, and
emits a fully-populated :class:`EntityState`. Behaviour, intention, prediction, and risk
are layered on top by later stages; here we establish *where everything is and how it is
moving*, in metres.
"""
from __future__ import annotations

import math

from ..config import CLASS_ALIASES, PEDESTRIAN_CLASSES, VEHICLE_CLASSES, CYCLIST_CLASSES
from ..schema import EntityState, EntityKind, BBox, Vec2, Behavior, Attention, VehicleManeuver
from ..worldmodel.homography import Homography
from ..worldmodel.motion import GroundTrack


def kind_of(cls: str) -> EntityKind:
    c = CLASS_ALIASES.get(cls, cls)
    if c in PEDESTRIAN_CLASSES:
        return EntityKind.PERSON
    if c in VEHICLE_CLASSES:
        return EntityKind.VEHICLE
    if c in CYCLIST_CLASSES:
        return EntityKind.CYCLIST
    return EntityKind.INFRASTRUCTURE


class WorldModel:
    """Maintains ground kinematics for all tracked entities across a clip."""

    def __init__(self, homography: Homography, dt: float, ground_bounds=None):
        self.H = homography
        self.dt = dt
        self.ground_bounds = ground_bounds
        self.tracks: dict[int, GroundTrack] = {}
        self.kinds: dict[int, EntityKind] = {}
        self.classes: dict[int, str] = {}

    def _project_foot(self, bbox: BBox):
        fx, fy = bbox.foot
        gx, gy = self.H.point_image_to_ground(fx, fy)
        return gx, gy

    def step(self, confirmed_tracks, frame: int, t: float) -> list[EntityState]:
        """Update ground kinematics from this frame's confirmed tracks; return states."""
        present = set()
        states: list[EntityState] = []
        for tk in confirmed_tracks:
            tid = tk.track_id
            present.add(tid)
            kind = self.kinds.get(tid) or kind_of(tk.cls)
            self.kinds[tid] = kind
            self.classes[tid] = tk.cls
            is_vehicle = kind in (EntityKind.VEHICLE, EntityKind.CYCLIST)
            gt = self.tracks.get(tid)
            if gt is None:
                gt = GroundTrack(tid, self.dt, is_vehicle=is_vehicle)
                self.tracks[tid] = gt
            gx, gy = self._project_foot(tk.bbox)
            gt.update((gx, gy), t, observed=True)

            v = gt.velocity
            state = EntityState(
                id=tid, kind=kind, cls=tk.cls, frame=frame, t=t,
                bbox=tk.bbox,
                position=Vec2(float(gt.position[0]), float(gt.position[1])),
                velocity=Vec2(float(v[0]), float(v[1])),
                acceleration=Vec2(float(gt.acceleration[0]), float(gt.acceleration[1])),
                heading=gt.heading, speed=gt.speed, jerk=gt.jerk,
                score=tk.score, visibility=1.0, age=gt.n,
                behavior=Behavior.UNKNOWN,
                attention=Attention.UNCERTAIN,
                maneuver=VehicleManeuver.CRUISE,
                attrs={},
            )
            if tk.keypoints:
                state.attrs["keypoints"] = tk.keypoints
            if getattr(tk, "last_gt_id", None) is not None:
                state.attrs["gt_id"] = tk.last_gt_id   # for evaluation only
            states.append(state)

        # coast the ground filters of entities not seen this frame (short occlusions)
        for tid, gt in self.tracks.items():
            if tid not in present:
                gt.coast(t)
        return states

    def get_track(self, tid: int) -> GroundTrack | None:
        return self.tracks.get(tid)
