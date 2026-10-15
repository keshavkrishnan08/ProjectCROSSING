"""The interaction engine, pair up entities and quantify every relationship.

Each frame, every vulnerable road user (pedestrian, cyclist) within a gating distance of a
vehicle becomes a candidate :class:`Interaction`. The engine computes the full conflict
geometry (:mod:`.metrics`) plus engineered context features (intention, signal state,
vehicle manoeuvre/braking, approach angle) and hands them to the risk engine. This is the
"who is interacting with whom" layer that turns a set of tracks into a graph of encounters.
"""
from __future__ import annotations

import math

from ..config import HORIZONS, PED_RADIUS, CYCLIST_WIDTH, VEHICLE_WIDTH
from ..schema import (EntityKind, EntityState, Interaction, RiskBand, WorldState,
                      VehicleManeuver)
from . import metrics as M

# effective circular footprint radius per class (metres)
_RADIUS = {"person": PED_RADIUS + 0.1, "cyclist": CYCLIST_WIDTH,
           "car": 1.6, "truck": 2.2, "van": 1.9, "bus": 2.6}


def footprint_radius(state: EntityState) -> float:
    return _RADIUS.get(state.cls, 1.6)


def _approach_angle(vru: EntityState, veh: EntityState) -> float:
    """Angle (rad) between the vehicle heading and the line of sight to the VRU (0=head-on)."""
    los = math.atan2(vru.position.y - veh.position.y, vru.position.x - veh.position.x)
    return abs(math.atan2(math.sin(los - veh.heading), math.cos(los - veh.heading)))


class InteractionEngine:
    def __init__(self, gate: float | None = None):
        self.gate = gate if gate is not None else HORIZONS.interaction_gate
        self._id = 0

    def step(self, ws: WorldState) -> list[Interaction]:
        ents = ws.entities
        vrus = [e for e in ents if e.kind in (EntityKind.PERSON, EntityKind.CYCLIST)]
        vehs = [e for e in ents if e.kind == EntityKind.VEHICLE]
        out: list[Interaction] = []
        for a in vrus:
            for b in vehs:
                d = M.distance(a.position, b.position)
                if d > self.gate:
                    continue
                radius = footprint_radius(a) + footprint_radius(b)
                m = M.all_metrics(a.position, a.velocity, b.position, b.velocity, radius)
                feats = self._features(a, b, m, radius)
                inter = Interaction(
                    id=self._id, frame=ws.frame, t=ws.t, a_id=a.id, b_id=b.id,
                    distance=m["distance"], closing_speed=m["closing_speed"],
                    rel_speed=m["rel_speed"], ttc=m["ttc"], dcpa=m["dcpa"],
                    tcpa=m["tcpa"], pet=m["pet"], path_conflict=m["path_conflict"],
                    band=RiskBand.LOW, features=feats,
                )
                self._id += 1
                out.append(inter)
                a.interaction_ids.append(inter.id)
                b.interaction_ids.append(inter.id)
        ws.interactions = out
        return out

    def _features(self, a: EntityState, b: EntityState, m: dict, radius: float) -> dict:
        """Engineered features for the risk model + explanation, all bounded/interpretable."""
        ttc = m["ttc"]
        inv_ttc = 0.0 if math.isinf(ttc) else 1.0 / max(ttc, 0.15)
        pet = m["pet"]
        inv_pet = 0.0 if math.isinf(pet) else 1.0 / max(pet, 0.15)
        veh_turning = b.maneuver in (VehicleManeuver.TURNING_LEFT, VehicleManeuver.TURNING_RIGHT)
        return {
            "distance": round(m["distance"], 3),
            "closing_speed": round(max(0.0, m["closing_speed"]), 3),
            "rel_speed": round(m["rel_speed"], 3),
            "inv_ttc": round(inv_ttc, 3),
            "dcpa": round(m["dcpa"], 3),
            "tcpa": round(max(0.0, m["tcpa"]), 3),
            "inv_pet": round(inv_pet, 3),
            "path_conflict": round(m["path_conflict"], 3),
            "veh_speed": round(b.speed, 3),
            "veh_braking": 1.0 if b.braking else 0.0,
            "veh_turning": 1.0 if veh_turning else 0.0,
            "vru_intention": round(a.crossing_intention, 3),
            "vru_speed": round(a.speed, 3),
            "vru_on_road": float(a.attrs.get("on_road", 0.0)),
            "vru_not_looking": 1.0 if a.attention.value == "not_looking" else 0.0,
            "approach_angle": round(_approach_angle(a, b), 3),
            "radius": round(radius, 3),
        }
