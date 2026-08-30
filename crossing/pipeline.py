"""End-to-end orchestration: detections in, a fully-populated :class:`Recording` out.

This is the spine that runs every stage in order and threads state between them:

    detections -> tracking -> world model -> behaviour (intention/attention)
               -> prediction -> interaction -> risk -> (post) near-miss events

Behavioural feature histories are maintained per pedestrian so the intention GRU always
sees a real 1.5 s window; vehicle manoeuvre/braking is inferred from ground kinematics;
signals are snapshotted per frame. After the timeline is built, the near-miss detector
aggregates interactions into events with early-warning times.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .config import Config, DEFAULT_CONFIG
from .schema import (Recording, WorldState, Scene, EntityKind, Behavior, Attention,
                     VehicleManeuver, SignalState)
from .worldmodel.homography import Homography
from .worldmodel.state import WorldModel
from .behavior.features import SceneContext, frame_features
from .behavior.intention import (IntentionModel, build_sequence, classify_behavior,
                                  classify_attention, OBS_FRAMES)
from .behavior import intention as _intent
from .perception.pose import gaze_lateral, looking_score
from .prediction.trajectory import TrajectoryPredictor
from .interaction.engine import InteractionEngine
from .risk.engine import RiskEngine
from .risk.nearmiss import NearMissDetector


class Pipeline:
    def __init__(self, scene: Scene, tracker, config: Config = DEFAULT_CONFIG,
                 predict_every: int = 3):
        self.scene = scene
        self.cfg = config
        self.H = Homography.from_list(scene.homography)
        self.dt = 1.0 / config.fps
        self.tracker = tracker
        self.world = WorldModel(self.H, self.dt, scene.ground_bounds)
        self.ctx = SceneContext(scene)
        self.intention = IntentionModel(use_learned=config.use_learned_intention)
        self.predictor = TrajectoryPredictor()
        self.interactions = InteractionEngine()
        self.risk = RiskEngine(use_learned=config.use_learned_risk)
        self.nearmiss = NearMissDetector()
        self.predict_every = predict_every
        self._feat_hist: dict[int, list] = defaultdict(list)

    # -- per-entity enrichment --------------------------------------------------------
    def _infer_vehicle_dynamics(self, e):
        gt = self.world.get_track(e.id)
        hx, hy = math.cos(e.heading), math.sin(e.heading)
        along = e.acceleration.x * hx + e.acceleration.y * hy
        e.braking = along < -1.6
        # heading rate from recent history
        turn_rate = 0.0
        if gt is not None and len(gt.vel_hist) >= 6:
            import numpy as np
            vs = list(gt.vel_hist)[-6:]
            h0 = math.atan2(vs[0][1], vs[0][0]); h1 = math.atan2(vs[-1][1], vs[-1][0])
            turn_rate = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)) / (5 * self.dt)
        if e.speed < 0.5:
            e.maneuver = VehicleManeuver.STOPPED
        elif e.braking:
            e.maneuver = VehicleManeuver.BRAKING
        elif turn_rate > 0.18:
            e.maneuver = VehicleManeuver.TURNING_LEFT
        elif turn_rate < -0.18:
            e.maneuver = VehicleManeuver.TURNING_RIGHT
        elif along > 0.8:
            e.maneuver = VehicleManeuver.ACCELERATING
        else:
            e.maneuver = VehicleManeuver.CRUISE

    def _enrich_pedestrian(self, e, others, frame):
        # pose-derived gaze / attention
        kps = e.attrs.get("keypoints")
        # which way is oncoming traffic relative to crossing? sign from nearest vehicle side
        cross_sign = 1.0
        gz = gaze_lateral(kps) if kps else 0.0
        e.attrs["gaze_cue"] = gz
        e.attrs["looking_score"] = looking_score(kps, cross_sign) if kps else None
        dist_curb = self.ctx.dist_curb(e.position.x, e.position.y)
        e.attrs["on_road"] = 1.0 if dist_curb < 0 else 0.0
        e.attrs["dist_curb"] = round(dist_curb, 2)

        feats = frame_features(e, others, self.ctx, frame)
        self._feat_hist[e.id].append(feats)
        if len(self._feat_hist[e.id]) > OBS_FRAMES + 4:
            self._feat_hist[e.id] = self._feat_hist[e.id][-(OBS_FRAMES + 4):]
        seq = build_sequence(self._feat_hist[e.id])
        p, std = self.intention.predict(seq, visibility=e.visibility)
        e.crossing_intention = p
        e.intention_std = std
        e.behavior = classify_behavior(e, dist_curb)
        e.attention = classify_attention(e)

    # -- main loop --------------------------------------------------------------------
    def process(self, detector, progress=None) -> Recording:
        n = detector.n_frames
        rec = Recording(scene=self.scene, fps=self.cfg.fps, meta={
            "tracker": type(self.tracker).__name__,
            "learned_intention": self.intention.is_learned,
            "learned_risk": self.risk.is_learned,
        })
        for f in range(n):
            t = f * self.dt
            dets = detector.detect(f)
            conf = self.tracker.update(dets, f)
            states = self.world.step(conf, f, t)

            peds = [s for s in states if s.kind in (EntityKind.PERSON, EntityKind.CYCLIST)]
            for e in states:
                if e.kind == EntityKind.VEHICLE:
                    self._infer_vehicle_dynamics(e)
            for e in peds:
                self._enrich_pedestrian(e, states, f)

            ws = WorldState(frame=f, t=t, entities=states)
            # predictions (subsample for cost; always predict active VRUs & near vehicles)
            if f % self.predict_every == 0:
                preds = []
                for e in states:
                    if e.kind in (EntityKind.PERSON, EntityKind.CYCLIST) or e.speed > 1.0:
                        preds.append(self.predictor.predict(e))
                ws.predictions = preds
            # interactions + risk
            inters = self.interactions.step(ws)
            self.risk.score_all(inters)
            # aggregate per-entity risk = max over its interactions this frame
            by_ent = defaultdict(float)
            for it in inters:
                by_ent[it.a_id] = max(by_ent[it.a_id], it.risk_score)
                by_ent[it.b_id] = max(by_ent[it.b_id], it.risk_score)
            for e in states:
                e.risk_score = round(by_ent.get(e.id, 0.0), 1)
            # signals snapshot
            ws.signals = {str(lt.id): lt.state_at(f).value for lt in self.scene.traffic_lights}
            rec.frames.append(ws)
            if progress and f % 60 == 0:
                progress(f, n)

        rec.events = self.nearmiss.detect(rec)
        rec.meta.update(self._summary(rec))
        return rec

    def _summary(self, rec: Recording) -> dict:
        ents = set()
        n_inter = 0
        for f in rec.frames:
            for e in f.entities:
                ents.add(e.id)
            n_inter += len(f.interactions)
        near = [e for e in rec.events if e.is_near_miss]
        ew = [e.early_warning_s for e in near if e.early_warning_s > 0]
        return {
            "n_frames": rec.n_frames,
            "duration_s": round(rec.duration, 2),
            "n_entities": len(ents),
            "n_interactions": n_inter,
            "n_events": len(rec.events),
            "n_near_misses": len(near),
            "mean_early_warning_s": round(sum(ew) / len(ew), 2) if ew else 0.0,
        }
