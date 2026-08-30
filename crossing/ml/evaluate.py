"""Comprehensive, honest evaluation across every stage of the stack.

Each capability is measured with its standard metric, on scenes the models never trained on:

* **detection** — precision / recall (via the simulator's ``gt_id`` back-reference).
* **tracking** — MOTA, IDF1, ID switches, fragmentation (real :class:`ByteTracker`).
* **trajectory** — minADE / minFDE / miss-rate over the multi-modal predictor.
* **intention** — lead time: how early P(cross) crosses 0.5 before road entry.

The risk benchmark lives in :mod:`crossing.ml.train_risk`; this module covers the rest.
"""
from __future__ import annotations

import math
from collections import defaultdict, Counter

import numpy as np

from ..config import FPS
from ..data.simulator import simulate
from ..tracking.bytetrack import ByteTracker
from ..tracking.oracle import OracleTracker
from ..perception.detector import ReplayDetector
from ..pipeline import Pipeline
from ..schema import EntityKind


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------
def evaluate_detection(sim) -> dict:
    tp = fp = fn = 0
    per_cls = defaultdict(lambda: [0, 0, 0])   # tp, fp, fn
    for f, dets in enumerate(sim.detections):
        gt_ids = {g["id"] for g in sim.gt_frames[f] if g.get("in_frame")}
        seen = set()
        for d in dets:
            if d.gt_id is not None and d.gt_id in gt_ids:
                tp += 1; per_cls[d.cls][0] += 1; seen.add(d.gt_id)
            else:
                fp += 1; per_cls[d.cls][1] += 1
        miss = gt_ids - seen
        fn += len(miss)
        for g in sim.gt_frames[f]:
            if g.get("in_frame") and g["id"] in miss:
                per_cls[g["cls"]][2] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    def _pc(v):
        t, f_, n_ = v
        return {
            "precision": round(t / (t + f_), 4) if (t + f_) else 0.0,
            "recall": round(t / (t + n_), 4) if (t + n_) else 0.0,
        }
    return {
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "per_class": {k: _pc(v) for k, v in per_cls.items()},
    }


# --------------------------------------------------------------------------------------
# tracking (MOTA / IDF1)
# --------------------------------------------------------------------------------------
def evaluate_tracking(sim) -> dict:
    tracker = ByteTracker()
    gt_total = 0
    fp = 0
    fn = 0
    idsw = 0
    gt_to_track = {}                # last track id assigned to each gt
    gt_track_frames = defaultdict(Counter)   # gt -> Counter(track_id -> frames)
    gt_ids_seen = defaultdict(set)
    for f in range(sim.n_frames):
        confirmed = tracker.update(sim.detections[f], f)
        gt_present = {g["id"] for g in sim.gt_frames[f] if g.get("in_frame")}
        gt_total += len(gt_present)
        matched_gt = set()
        for t in confirmed:
            g = t.last_gt_id
            if g is None or g not in gt_present:
                fp += 1
                continue
            matched_gt.add(g)
            gt_track_frames[g][t.track_id] += 1
            gt_ids_seen[g].add(t.track_id)
            if g in gt_to_track and gt_to_track[g] != t.track_id:
                idsw += 1
            gt_to_track[g] = t.track_id
        fn += len(gt_present - matched_gt)

    # IDF1 via dominant assignment
    idtp = sum(cnt.most_common(1)[0][1] for cnt in gt_track_frames.values() if cnt)
    total_track_frames = sum(sum(c.values()) for c in gt_track_frames.values())
    idp = idtp / max(total_track_frames, 1)
    idr = idtp / max(gt_total, 1)
    idf1 = 2 * idp * idr / (idp + idr) if (idp + idr) else 0.0
    mota = 1.0 - (fn + fp + idsw) / max(gt_total, 1)
    frags = [len(s) for s in gt_ids_seen.values()]
    return {
        "mota": round(mota, 4), "idf1": round(idf1, 4),
        "id_switches": idsw, "fragmentation": round(float(np.mean(frags)), 2) if frags else 0.0,
        "fp": fp, "fn": fn, "gt_total": gt_total,
        "gt_coverage": round(len(gt_ids_seen) / max(len({g['id'] for gf in sim.gt_frames for g in gf if g.get('in_frame')}), 1), 3),
    }


# --------------------------------------------------------------------------------------
# trajectory (minADE / minFDE)
# --------------------------------------------------------------------------------------
def evaluate_trajectory(rec, horizons=(1.0, 2.0, 3.0)) -> dict:
    # index actual positions by (id, frame)
    pos = {}
    for ws in rec.frames:
        for e in ws.entities:
            pos[(e.id, ws.frame)] = (e.position.x, e.position.y)
    ade_ped, fde_ped, miss_ped = [], [], []
    for ws in rec.frames:
        for pred in ws.predictions:
            eid = pred.entity_id
            if not pred.modes:
                continue
            # only score pedestrians (their multimodality is the interesting case)
            ekinds = [e.kind for e in ws.entities if e.id == eid]
            if not ekinds or ekinds[0] != EntityKind.PERSON:
                continue
            mode_ades = []
            top = max(pred.modes, key=lambda m: m.prob)
            fdes = []
            for mode in pred.modes:
                errs = []
                for (t, px, py) in mode.points:
                    if t > max(horizons) + 1e-6:
                        continue
                    fk = ws.frame + int(round(t * FPS))
                    actual = pos.get((eid, fk))
                    if actual is None:
                        continue
                    errs.append(math.hypot(px - actual[0], py - actual[1]))
                if errs:
                    mode_ades.append(np.mean(errs))
                    fdes.append(errs[-1])
            if mode_ades:
                ade_ped.append(min(mode_ades))       # minADE over modes
                fde_ped.append(min(fdes))
                miss_ped.append(1.0 if min(fdes) > 2.0 else 0.0)
    return {
        "minADE_m": round(float(np.mean(ade_ped)), 3) if ade_ped else None,
        "minFDE_m": round(float(np.mean(fde_ped)), 3) if fde_ped else None,
        "miss_rate@2m": round(float(np.mean(miss_ped)), 3) if miss_ped else None,
        "n": len(ade_ped),
    }


# --------------------------------------------------------------------------------------
# intention lead time
# --------------------------------------------------------------------------------------
def evaluate_intention_leadtime(sim, rec) -> dict:
    from ..behavior.features import SceneContext
    ctx = SceneContext(sim.scene)
    # ground-truth road-entry frame per gt ped
    entry = {}
    for f, gframe in enumerate(sim.gt_frames):
        for g in gframe:
            if g["kind"] == "person" and ctx.dist_curb(g["x"], g["y"]) < 0:
                entry.setdefault(g["id"], f)
    # per reconstructed ped, first frame P(cross)>=0.5
    first_high = {}
    for ws in rec.frames:
        for e in ws.entities:
            if e.kind != EntityKind.PERSON:
                continue
            gid = e.attrs.get("gt_id")
            if gid is None:
                continue
            if e.crossing_intention >= 0.5 and gid not in first_high:
                first_high[gid] = ws.frame
    leads = []
    detected = 0
    for gid, fe in entry.items():
        fh = first_high.get(gid)
        if fh is not None and fh <= fe:
            leads.append((fe - fh) / FPS)
            detected += 1
    return {
        "mean_lead_time_s": round(float(np.mean(leads)), 2) if leads else 0.0,
        "median_lead_time_s": round(float(np.median(leads)), 2) if leads else 0.0,
        "recall_before_entry": round(detected / max(len(entry), 1), 3),
        "n_crossings": len(entry),
    }


def evaluate_all(seeds, duration=30.0, use_learned=True) -> dict:
    det, trk, traj, intent = [], [], [], []
    from ..config import Config
    cfg = Config(use_learned_intention=use_learned, use_learned_risk=use_learned)
    for seed in seeds:
        sim = simulate(seed=seed, duration_s=duration)
        det.append(evaluate_detection(sim))
        trk.append(evaluate_tracking(sim))
        rec = Pipeline(sim.scene, OracleTracker(), cfg).process(ReplayDetector(sim.detections))
        traj.append(evaluate_trajectory(rec))
        intent.append(evaluate_intention_leadtime(sim, rec))

    def _avg(dicts, keys):
        out = {}
        for k in keys:
            vals = [d[k] for d in dicts if d.get(k) is not None]
            out[k] = round(float(np.mean(vals)), 4) if vals else None
        return out

    return {
        "detection": _avg(det, ["precision", "recall", "f1"]),
        "tracking": _avg(trk, ["mota", "idf1", "id_switches", "fragmentation", "gt_coverage"]),
        "trajectory": _avg(traj, ["minADE_m", "minFDE_m", "miss_rate@2m"]),
        "intention": _avg(intent, ["mean_lead_time_s", "median_lead_time_s", "recall_before_entry"]),
        "n_scenes": len(seeds),
    }
