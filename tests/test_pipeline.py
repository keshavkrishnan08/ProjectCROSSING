"""Integration + component smoke tests over the simulator and full pipeline."""
import math
from collections import defaultdict

import numpy as np

from crossing.data.simulator import simulate
from crossing.tracking.oracle import OracleTracker
from crossing.tracking.kalman import BBoxKalman, GroundKalman
from crossing.perception.detector import ReplayDetector
from crossing.pipeline import Pipeline
from crossing.config import Config
from crossing.schema import Recording


def test_schema_recording_roundtrip():
    sim = simulate(seed=5, duration_s=6.0)
    rec = Pipeline(sim.scene, OracleTracker(), Config()).process(ReplayDetector(sim.detections))
    d = rec.to_dict()
    rec2 = Recording.from_dict(d)
    assert rec2.n_frames == rec.n_frames
    assert len(rec2.events) == len(rec.events)
    assert rec2.scene.name == rec.scene.name


def test_ground_kalman_tracks_constant_velocity():
    kf = GroundKalman(dt=1 / 30.0).initiate((0.0, 0.0))
    for i in range(60):
        kf.predict()
        kf.update((0.1 * i + np.random.normal(0, 0.02), 0.0))
    assert abs(kf.velocity[0] - 3.0) < 0.6      # 0.1 m/frame * 30 fps = 3 m/s
    assert abs(kf.velocity[1]) < 0.5


def test_bbox_kalman_smooths():
    from crossing.schema import BBox
    from crossing.tracking.bytetrack import _xyah, _xyah_to_bbox
    kf = BBoxKalman()
    m, c = kf.initiate(_xyah(BBox(100, 400, 140, 500)))
    for i in range(1, 15):
        m, c = kf.predict(m, c)
        m, c = kf.update(m, c, _xyah(BBox(100 + 5 * i, 400, 140 + 5 * i, 500)))
    est = _xyah_to_bbox(m[:4])
    assert abs(est.cx - (120 + 5 * 14)) < 3 and abs(est.h - 100) < 3


def test_simulator_scripted_conflicts_are_close():
    sim = simulate(seed=7, duration_s=40.0)
    idx = [{x["id"]: x for x in g} for g in sim.gt_frames]
    tight = 0
    for c in sim.gt_conflicts:
        dmin = min((math.hypot(fr[c["veh"]]["x"] - fr[c["ped"]]["x"],
                               fr[c["veh"]]["y"] - fr[c["ped"]]["y"])
                    for fr in idx if c["veh"] in fr and c["ped"] in fr), default=99)
        tight += dmin < 2.5
    assert tight == len(sim.gt_conflicts)      # every scripted conflict is a genuine close call


def test_pipeline_detects_near_misses_matching_ground_truth():
    from crossing.behavior.features import SceneContext
    sim = simulate(seed=7, duration_s=40.0)
    rec = Pipeline(sim.scene, OracleTracker(), Config()).process(ReplayDetector(sim.detections))
    ctx = SceneContext(sim.scene)
    # ground‑truth close encounters (veh<->ped centre < 1.2 m)
    pm = defaultdict(lambda: 1e9)
    for gf in sim.gt_frames:
        fr = {x["id"]: x for x in gf if x.get("in_frame")}
        vs = [x for x in fr.values() if x["kind"] == "vehicle"]
        ps = [x for x in fr.values() if x["kind"] == "person"]
        for v in vs:
            for p in ps:
                pm[(v["id"], p["id"])] = min(pm[(v["id"], p["id"])],
                                             math.hypot(v["x"] - p["x"], v["y"] - p["y"]))
    gt_close = sum(1 for d in pm.values() if d < 1.2)
    detected = sum(1 for e in rec.events if e.is_near_miss)
    # detector should be in the same ballpark as ground truth (not 0, not 10x)
    assert gt_close > 0
    assert 0.4 * gt_close <= detected <= 2.5 * gt_close + 3


def test_world_model_reconstruction_accuracy():
    from crossing.worldmodel.state import WorldModel
    from crossing.worldmodel.homography import Homography
    sim = simulate(seed=11, duration_s=15.0)
    H = Homography.from_list(sim.scene.homography)
    wm = WorldModel(H, dt=1 / sim.fps, ground_bounds=sim.scene.ground_bounds)
    tr = OracleTracker()
    errs = []
    for f in range(sim.n_frames):
        states = wm.step(tr.update(sim.detections[f], f), f, f / sim.fps)
        gt = {x["id"]: x for x in sim.gt_frames[f]}
        for s in states:
            gid = s.attrs.get("gt_id")
            if gid in gt and gt[gid]["in_frame"]:
                errs.append(math.hypot(s.position.x - gt[gid]["x"], s.position.y - gt[gid]["y"]))
    assert np.median(errs) < 1.0     # sub‑metre median ground reconstruction
