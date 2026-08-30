"""Dataset construction for the learned models — with leakage-free, by-scene splits.

The cardinal sin in trajectory/intention work is splitting individual frames randomly, so
the same pedestrian appears in train and test. CROSSING never does that: each simulated
scene is an independent unit (deterministic in its seed), and splits are *by scene*.

Two datasets are built, both with **features from the reconstructed pipeline** (perception
-> tracking -> world model, exactly what runs at inference) and **labels from ground
truth** (aligned via each track's ``gt_id``):

* **intention** — per-pedestrian temporal sequences; label = does the pedestrian enter the
  roadway within the next second? (an *early* prediction task, not hindsight).
* **risk** — per-interaction feature vectors; label = does this pair reach a genuine
  near-miss within the next ~2.5 s? (a near-miss *precursor* label).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import FPS, RISK
from ..schema import EntityKind
from ..data.simulator import simulate
from ..tracking.oracle import OracleTracker
from ..perception.detector import ReplayDetector
from ..pipeline import Pipeline
from ..behavior.features import SceneContext, frame_features
from ..behavior.intention import build_sequence, OBS_FRAMES
from ..risk.engine import RISK_FEATURES


_SCENE_CACHE: dict = {}


def generate_scene(seed: int, duration: float = 30.0, use_learned_risk: bool = False):
    """Simulate a scene and run the full pipeline; return (sim_result, recording).

    Cached by (seed, duration) so the intention and risk dataset builders reuse a single
    simulation instead of re-running it. Risk-dataset labels come from realised geometry,
    so we build datasets with the *physics* risk (``use_learned_risk=False``) to avoid a
    trained model leaking into its own training features.
    """
    key = (seed, round(duration, 3), use_learned_risk)
    if key in _SCENE_CACHE:
        return _SCENE_CACHE[key]
    from ..config import Config
    cfg = Config(use_learned_risk=use_learned_risk, use_learned_intention=False)
    sim = simulate(seed=seed, duration_s=duration)
    rec = Pipeline(sim.scene, OracleTracker(), cfg).process(ReplayDetector(sim.detections))
    if len(_SCENE_CACHE) < 40:
        _SCENE_CACHE[key] = (sim, rec)
    return sim, rec


def _gt_onroad_timeline(sim, ctx: SceneContext):
    """Per gt_id -> dict(frame -> on_road bool) from ground-truth positions."""
    out = defaultdict(dict)
    for f, gframe in enumerate(sim.gt_frames):
        for g in gframe:
            if g["kind"] == "person":
                out[g["id"]][f] = ctx.dist_curb(g["x"], g["y"]) < 0.0
    return out


def intention_samples(sim, rec, ctx, horizon_frames=30, stride=3):
    """Yield (sequence[T,D], label, gt_id) for pedestrians across a scene."""
    onroad = _gt_onroad_timeline(sim, ctx)
    hist = defaultdict(list)
    samples = []
    for ws in rec.frames:
        others = ws.entities
        for e in ws.entities:
            if e.kind != EntityKind.PERSON:
                continue
            feats = frame_features(e, others, ctx, ws.frame)
            hist[e.id].append(feats)
            if len(hist[e.id]) > OBS_FRAMES + 4:
                hist[e.id] = hist[e.id][-(OBS_FRAMES + 4):]
            gid = e.attrs.get("gt_id")
            if gid is None or gid not in onroad:
                continue
            # label: does GT enter road within the next `horizon_frames`?
            fut = [onroad[gid].get(ws.frame + k, False) for k in range(1, horizon_frames + 1)]
            now_on = onroad[gid].get(ws.frame, False)
            label = 1 if (any(fut) and not now_on) else (1 if now_on else 0)
            # sample on a stride and require a little history
            if ws.frame % stride == 0 and len(hist[e.id]) >= 4:
                seq = build_sequence(hist[e.id])
                samples.append((seq, label, gid))
    return samples


def _pair_timeline(rec):
    """(a_id,b_id) -> sorted list of (frame, distance, closing_speed)."""
    tl = defaultdict(list)
    for ws in rec.frames:
        for it in ws.interactions:
            tl[(it.a_id, it.b_id)].append((it.frame, it.distance, it.closing_speed))
    for k in tl:
        tl[k].sort()
    return tl


def risk_samples(rec, window_s=2.5, stride=2):
    """Yield (features_dict, label) for interactions; label = near-miss precursor.

    The label uses the SAME operational definition as
    :class:`crossing.risk.nearmiss.NearMissDetector`: within the lookahead window the pair
    must reach a small realised separation *with* real closing speed, AND the vulnerable
    road user must actually be in / entering the roadway (a car passing a curbside
    pedestrian is not a near-miss). Aligning train and eval keeps the risk model's
    calibration claim honest.
    """
    W = int(window_s * FPS)
    tl = _pair_timeline(rec)
    pair_dist = {k: {fr: (d, c) for fr, d, c in v} for k, v in tl.items()}
    # per-(frame, vru_id) roadway membership from the reconstructed states
    onroad = {}
    for ws in rec.frames:
        for e in ws.entities:
            if e.kind in (EntityKind.PERSON, EntityKind.CYCLIST):
                dc = e.attrs.get("dist_curb", 9.0)
                onroad[(ws.frame, e.id)] = (e.attrs.get("on_road", 0.0) > 0.5) or \
                                           (dc < 0.5 and e.speed > 0.4)
    samples = []
    for ws in rec.frames:
        if ws.frame % stride != 0:
            continue
        for it in ws.interactions:
            fut = pair_dist.get((it.a_id, it.b_id), {})
            frames = [f for f in range(it.frame, it.frame + W) if f in fut]
            realized_min = min((fut[f][0] for f in frames), default=99.0)
            max_close = max((fut[f][1] for f in frames), default=0.0)
            in_conflict = any(onroad.get((f, it.a_id), False) for f in frames)
            label = 1 if (in_conflict and realized_min < RISK.nearmiss_dcpa
                          and max_close >= RISK.nearmiss_min_closing_speed) else 0
            samples.append((it.features, label))
    return samples


def build_intention_dataset(seeds, duration=30.0, verbose=False):
    X, y, groups = [], [], []
    for si, seed in enumerate(seeds):
        sim, rec = generate_scene(seed, duration)
        ctx = SceneContext(sim.scene)
        for seq, label, gid in intention_samples(sim, rec, ctx):
            X.append(seq); y.append(label); groups.append(si)
        if verbose:
            print(f"  scene seed={seed}: cumulative samples={len(X)}")
    return np.array(X), np.array(y), np.array(groups)


def build_risk_dataset(seeds, duration=30.0, verbose=False):
    rows, y, groups, feat_dicts = [], [], [], []
    for si, seed in enumerate(seeds):
        sim, rec = generate_scene(seed, duration)
        for feats, label in risk_samples(rec):
            rows.append([feats.get(k, 0.0) for k in RISK_FEATURES])
            feat_dicts.append(feats)
            y.append(label); groups.append(si)
        if verbose:
            print(f"  scene seed={seed}: cumulative samples={len(rows)}")
    return np.array(rows, float), np.array(y), np.array(groups), feat_dicts
