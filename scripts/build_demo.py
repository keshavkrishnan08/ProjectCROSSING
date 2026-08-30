#!/usr/bin/env python3
"""Build the compact, self-contained data bundle the CROSSING web demo embeds.

Runs the full pipeline on a showcase scene (chosen for a clean spread of conflict types),
attaches counterfactual simulations for the top near-misses, aggregates a city-scale risk
map across several scenes, and writes everything to a single JSON small enough to inline
into the Artifact (frames subsampled to ~15 fps, coordinates rounded, predictions trimmed).
"""
from __future__ import annotations

import argparse
import json
import math

from crossing.config import Config, RECORDINGS_DIR, METRICS_DIR
from crossing.data.simulator import simulate
from crossing.tracking.oracle import OracleTracker
from crossing.perception.detector import ReplayDetector
from crossing.pipeline import Pipeline
from crossing.risk.counterfactual import CounterfactualEngine
from crossing.risk.explain import decompose, rationale
from crossing.graph.eventgraph import CityAnalyzer, build_event_graph, graph_summary


def compact_recording(rec, stride=2, pred_ids=None):
    """Export the recording as a compact dict: subsampled frames, trimmed predictions."""
    scene = rec.scene
    frames = []
    for i, ws in enumerate(rec.frames):
        if i % stride != 0:
            continue
        ents = []
        for e in ws.entities:
            d = {
                "id": e.id, "k": e.kind.value[0], "cls": e.cls,
                "p": [round(e.position.x, 2), round(e.position.y, 2)],
                "v": [round(e.velocity.x, 2), round(e.velocity.y, 2)],
                "hd": round(e.heading, 3), "sp": round(e.speed, 2),
                "bb": [round(v) for v in e.bbox.to_list()] if e.bbox else None,
                "beh": e.behavior.value, "att": e.attention.value,
                "man": e.maneuver.value, "int": round(e.crossing_intention, 3),
                "istd": round(e.intention_std, 3), "brk": int(e.braking),
                "risk": round(e.risk_score, 1),
                "onroad": int(e.attrs.get("on_road", 0)),
            }
            ents.append(d)
        inters = [{
            "a": it.a_id, "b": it.b_id, "d": round(it.distance, 2),
            "ttc": None if math.isinf(it.ttc) else round(it.ttc, 2),
            "dcpa": round(it.dcpa, 2), "risk": round(it.risk_score, 1),
            "band": it.band.value,
        } for it in ws.interactions if it.risk_score >= 25 or it.distance < 6]
        preds = []
        for pr in ws.predictions:
            if pred_ids is not None and pr.entity_id not in pred_ids:
                continue
            preds.append({
                "id": pr.entity_id,
                "m": [{"p": round(m.prob, 2), "l": m.label,
                       "pts": [[round(pt[0], 1), round(pt[1], 2), round(pt[2], 2)] for pt in m.points]}
                      for m in pr.modes],
            })
        frames.append({"f": ws.frame, "t": round(ws.t, 2), "e": ents,
                       "i": inters, "pr": preds, "sig": ws.signals})
    events = [ev.to_dict() for ev in rec.events]
    return {
        "scene": scene.to_dict(),
        "fps": rec.fps, "stride": stride,
        "frames": frames, "events": events, "meta": rec.meta,
    }


def build(showcase_seed=7, duration=45.0, city_seeds=None, out=None):
    city_seeds = city_seeds or [7, 11, 21, 33, 42, 51, 63, 77, 88, 99]
    cfg = Config(use_learned_intention=True, use_learned_risk=True)

    print(f"[demo] showcase scene seed={showcase_seed} ({duration}s)")
    sim = simulate(seed=showcase_seed, duration_s=duration)
    rec = Pipeline(sim.scene, OracleTracker(), cfg).process(ReplayDetector(sim.detections))

    # entities that ever participate in a near-miss / high event get predicted trajectories kept
    keep_pred = set()
    near = [e for e in rec.events if e.is_near_miss]
    top_events = sorted(rec.events, key=lambda e: (not e.is_near_miss, -e.peak_risk))[:6]
    for ev in top_events:
        keep_pred.add(ev.a_id); keep_pred.add(ev.b_id)

    bundle = compact_recording(rec, stride=2, pred_ids=None)

    # counterfactuals for the top near-miss events
    cf = CounterfactualEngine()
    cfs = {}
    for ev in top_events:
        cfs[str(ev.id)] = cf.for_event(rec, ev)
    bundle["counterfactuals"] = cfs

    # explanations
    for ev in bundle["events"]:
        rows = decompose(ev.get("contributions", {}))
        ev["explanation"] = [{"factor": r[0], "points": round(r[1], 1), "share": round(r[2], 3)} for r in rows]
        ev["rationale"] = rationale(ev.get("contributions", {}), ev.get("peak_risk", 0))

    # knowledge-graph summary for the showcase
    try:
        bundle["graph"] = graph_summary(build_event_graph(rec))
    except Exception:
        bundle["graph"] = None

    # ---- city-scale aggregation across many scenes ----
    print(f"[demo] aggregating city view across {len(city_seeds)} scenes...")
    ca = CityAnalyzer(sim.scene.ground_bounds, cell=3.0)
    ca.add(rec)
    for s in city_seeds:
        if s == showcase_seed:
            continue
        ssim = simulate(seed=s, duration_s=duration)
        srec = Pipeline(ssim.scene, OracleTracker(), cfg).process(ReplayDetector(ssim.detections))
        ca.add(srec)
    bundle["city"] = ca.summary(top=8)

    # attach training metrics if present
    metrics = {}
    for name in ("summary", "risk", "intention", "capabilities"):
        p = METRICS_DIR / f"{name}.json"
        if p.exists():
            metrics[name] = json.loads(p.read_text())
    bundle["metrics"] = metrics

    out = out or (RECORDINGS_DIR / "demo_scene.json")
    out.write_text(json.dumps(bundle, separators=(",", ":")))
    size_mb = out.stat().st_size / 1e6
    print(f"[demo] wrote {out}  ({size_mb:.2f} MB)")
    print(f"[demo] frames={len(bundle['frames'])} events={len(bundle['events'])} "
          f"near_misses={len(near)} city_hotspots={len(bundle['city']['hotspots'])}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=45.0)
    build(showcase_seed=ap.parse_args().seed, duration=ap.parse_args().duration)
