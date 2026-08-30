"""Entity–event knowledge graph and city-scale aggregation.

Beyond individual frames, CROSSING represents a scene the way an intelligence platform
does: real-world **entities** (people, vehicles, infrastructure), the **events** they
produce, and the **relationships** between them —

    PERSON_17 --participates_in--> INTERACTION_1821 --involves--> VEHICLE_08
    INTERACTION_1821 --occurs_at--> "E crosswalk" --produces--> NEAR_MISS_918

:func:`build_event_graph` materialises that graph (NetworkX) for one recording.
:class:`CityAnalyzer` rolls many recordings up into a spatial risk map, ranked hotspots,
and systemic patterns (which configurations predict danger) — the "where is the environment
systematically unsafe?" view.
"""
from __future__ import annotations

import math
from collections import defaultdict, Counter

import numpy as np

try:
    import networkx as nx
except Exception:                 # pragma: no cover
    nx = None

from ..schema import Recording, RiskEvent, VehicleManeuver


def build_event_graph(rec: Recording):
    """Materialise a NetworkX knowledge graph for a recording."""
    if nx is None:
        raise RuntimeError("networkx not available")
    G = nx.MultiDiGraph()
    # infrastructure + zones
    for cw in rec.scene.crosswalks:
        G.add_node(f"crosswalk:{cw.id}", kind="infrastructure", type="crosswalk")
    for lt in rec.scene.traffic_lights:
        G.add_node(f"light:{lt.id}", kind="infrastructure", type="traffic_light",
                   controls=lt.controls)
    # entities (dedup by id)
    ent_kind = {}
    for f in rec.frames:
        for e in f.entities:
            ent_kind[e.id] = e.kind.value
    for eid, k in ent_kind.items():
        G.add_node(f"entity:{eid}", kind=k)
    # events
    for ev in rec.events:
        n = f"event:{ev.id}"
        G.add_node(n, kind="event", band=ev.band.value, near_miss=ev.is_near_miss,
                   peak_risk=ev.peak_risk, zone=ev.zone, label=ev.label,
                   t=ev.peak_t, early_warning=ev.early_warning_s)
        G.add_edge(f"entity:{ev.a_id}", n, rel="participates_in")
        G.add_edge(n, f"entity:{ev.b_id}", rel="involves")
        if ev.zone:
            zn = f"zone:{ev.zone}"
            if zn not in G:
                G.add_node(zn, kind="zone")
            G.add_edge(n, zn, rel="occurs_at")
    return G


def graph_summary(G) -> dict:
    kinds = Counter(d.get("kind") for _, d in G.nodes(data=True))
    rels = Counter(d.get("rel") for *_ , d in G.edges(data=True))
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "node_kinds": dict(kinds), "relations": dict(rels)}


class CityAnalyzer:
    """Aggregates many recordings (time windows / locations) into safety intelligence."""

    def __init__(self, ground_bounds, cell: float = 3.0):
        self.bounds = ground_bounds
        self.cell = cell
        xmin, ymin, xmax, ymax = ground_bounds
        self.nx = max(1, int(math.ceil((xmax - xmin) / cell)))
        self.ny = max(1, int(math.ceil((ymax - ymin) / cell)))
        self.grid_risk = np.zeros((self.ny, self.nx))
        self.grid_count = np.zeros((self.ny, self.nx))
        self.grid_nearmiss = np.zeros((self.ny, self.nx))
        self.zone_stats = defaultdict(lambda: {"events": 0, "near_misses": 0,
                                               "risk_sum": 0.0, "causes": Counter()})
        self.total_events = 0
        self.total_near_misses = 0
        self.total_interactions = 0
        self.early_warnings = []
        self._exposure = 0  # entity-frames, a denominator for normalisation

    def _cell_of(self, x, y):
        xmin, ymin, xmax, ymax = self.bounds
        j = int(np.clip((x - xmin) / self.cell, 0, self.nx - 1))
        i = int(np.clip((y - ymin) / self.cell, 0, self.ny - 1))
        return i, j

    def add(self, rec: Recording):
        self.total_interactions += sum(len(f.interactions) for f in rec.frames)
        self._exposure += sum(len(f.entities) for f in rec.frames)
        for ev in rec.events:
            self.total_events += 1
            zs = self.zone_stats[ev.zone or "unzoned"]
            zs["events"] += 1
            zs["risk_sum"] += ev.peak_risk
            zs["causes"][self._primary_cause(ev)] += 1
            if ev.is_near_miss:
                self.total_near_misses += 1
                zs["near_misses"] += 1
                if ev.early_warning_s > 0:
                    self.early_warnings.append(ev.early_warning_s)
            if ev.location:
                i, j = self._cell_of(ev.location[0], ev.location[1])
                self.grid_risk[i, j] += ev.peak_risk
                self.grid_count[i, j] += 1
                self.grid_nearmiss[i, j] += int(ev.is_near_miss)

    @staticmethod
    def _primary_cause(ev: RiskEvent) -> str:
        if "Turning" in ev.label:
            return "left/right turn conflict"
        if ev.evasive.get("vehicle_hard_braking"):
            return "late braking"
        if ev.min_ttc is not None and not math.isinf(ev.min_ttc) and ev.min_ttc < 1.0:
            return "short gap / speed"
        if "Mid-crossing" in ev.label:
            return "mid-crossing exposure"
        return "approach conflict"

    def hotspots(self, top: int = 10) -> list[dict]:
        rows = []
        for zone, s in self.zone_stats.items():
            rows.append({
                "zone": zone,
                "events": s["events"],
                "near_misses": s["near_misses"],
                "mean_risk": round(s["risk_sum"] / max(s["events"], 1), 1),
                "near_miss_rate": round(s["near_misses"] / max(s["events"], 1), 3),
                "primary_cause": (s["causes"].most_common(1)[0][0] if s["causes"] else "-"),
                "causes": dict(s["causes"]),
            })
        rows.sort(key=lambda r: (-r["near_misses"], -r["mean_risk"]))
        return rows[:top]

    def risk_grid(self) -> dict:
        """Mean-risk and near-miss grids for the spatial heatmap."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_risk = np.where(self.grid_count > 0, self.grid_risk / self.grid_count, 0.0)
        return {
            "nx": self.nx, "ny": self.ny, "cell": self.cell, "bounds": list(self.bounds),
            "mean_risk": np.round(mean_risk, 1).tolist(),
            "count": self.grid_count.astype(int).tolist(),
            "near_miss": self.grid_nearmiss.astype(int).tolist(),
        }

    def patterns(self) -> dict:
        causes = Counter()
        for s in self.zone_stats.values():
            causes.update(s["causes"])
        return {
            "total_events": self.total_events,
            "total_near_misses": self.total_near_misses,
            "total_interactions": self.total_interactions,
            "exposure_entity_frames": self._exposure,
            "near_miss_per_1k_interactions": round(
                1000 * self.total_near_misses / max(self.total_interactions, 1), 3),
            "mean_early_warning_s": round(float(np.mean(self.early_warnings)), 2)
            if self.early_warnings else 0.0,
            "cause_breakdown": dict(causes.most_common()),
        }

    def summary(self, top=10) -> dict:
        return {"hotspots": self.hotspots(top), "grid": self.risk_grid(),
                "patterns": self.patterns()}
