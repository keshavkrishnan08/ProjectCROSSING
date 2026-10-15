"""A ByteTrack-style multi-object tracker.

Detection turns a frame into boxes; tracking turns a *stream* of boxes into **persistent
identities** PERSON_17 from t=0 to t=401, even across the moment a car briefly hides
them. That persistence is what makes trajectories, intention ramps, and interaction
histories possible.

Implements the essential ideas of ByteTrack (Zhang et al., 2022):

* a per-track :class:`BBoxKalman` motion model;
* **two-stage association** match high-confidence detections first, then recover
  low-confidence ones against still-unmatched tracks (this is what keeps IDs alive through
  partial occlusion, where a real detector's score dips rather than vanishes);
* IoU association fused with a Mahalanobis motion gate;
* a lost-track buffer so an identity survives a short full occlusion and is re-linked.

Pure NumPy + scipy's linear-sum assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..schema import Detection, BBox
from .kalman import BBoxKalman


def _xyah(b: BBox) -> np.ndarray:
    """BBox -> [cx, cy, aspect, height]."""
    return np.array([b.cx, b.cy, b.w / max(b.h, 1e-6), b.h])


def _xyah_to_bbox(z: np.ndarray) -> BBox:
    cx, cy, a, h = z
    w = a * h
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou(b1: BBox, b2: BBox) -> float:
    return b1.iou(b2)


@dataclass
class Track:
    track_id: int
    cls: str
    mean: np.ndarray
    cov: np.ndarray
    kf: BBoxKalman
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "tentative"          # tentative | confirmed | lost
    score: float = 0.5
    keypoints: list = None
    last_gt_id: int = None            # carried through only for evaluation convenience
    history: list = field(default_factory=list)   # list of (frame, BBox)

    @property
    def bbox(self) -> BBox:
        return _xyah_to_bbox(self.mean[:4])

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, _xyah(det.bbox))
        self.hits += 1
        self.time_since_update = 0
        self.score = 0.7 * self.score + 0.3 * det.score
        self.keypoints = det.keypoints
        self.last_gt_id = det.gt_id


class ByteTracker:
    """Online multi-object tracker with two-stage (high/low score) association."""

    def __init__(self, high_thresh=0.5, low_thresh=0.2, match_thresh=0.8,
                 max_age=45, n_init=3, gate=100.0):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh   # 1 - IoU distance acceptance
        self.max_age = max_age
        self.n_init = n_init
        self.gate = gate                   # Mahalanobis chi2 gate (4 dof ~9.49 at .95; loose here)
        self._next_id = 1
        self.tracks: list[Track] = []
        self.frame = -1

    # -- association helpers ----------------------------------------------------------
    def _match(self, tracks, dets):
        """Hungarian match tracks<->dets by IoU, motion-gated. Returns matches, u_t, u_d."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou_mat = np.zeros((len(tracks), len(dets)))
        det_xyah = np.array([_xyah(d.bbox) for d in dets])
        for ti, t in enumerate(tracks):
            gd = t.kf.gating_distance(t.mean, t.cov, det_xyah)
            for di, d in enumerate(dets):
                iou = _iou(t.bbox, d.bbox)
                if gd[di] > self.gate or t.cls != d.cls:
                    iou = 0.0
                iou_mat[ti, di] = iou
        cost = 1.0 - iou_mat
        rows, cols = linear_sum_assignment(cost)
        matches, u_t, u_d = [], [], []
        assigned_t, assigned_d = set(), set()
        for r, c in zip(rows, cols):
            if iou_mat[r, c] >= (1.0 - self.match_thresh):
                matches.append((r, c)); assigned_t.add(r); assigned_d.add(c)
        u_t = [i for i in range(len(tracks)) if i not in assigned_t]
        u_d = [i for i in range(len(dets)) if i not in assigned_d]
        return matches, u_t, u_d

    # -- main step --------------------------------------------------------------------
    def update(self, detections: list[Detection], frame: int) -> list[Track]:
        self.frame = frame
        for t in self.tracks:
            t.predict()

        dets = [d for d in detections if d.cls != "traffic_light"]
        high = [d for d in dets if d.score >= self.high_thresh]
        low = [d for d in dets if self.low_thresh <= d.score < self.high_thresh]

        active = [t for t in self.tracks if t.state != "lost"]
        lost = [t for t in self.tracks if t.state == "lost"]

        # --- stage 1: high-score dets vs active + lost tracks (lost can be re-activated) ---
        pool = active + lost
        m1, ut1, ud1 = self._match(pool, high)
        for ti, di in m1:
            pool[ti].update(high[di])
            pool[ti].state = "confirmed" if pool[ti].hits >= self.n_init else "tentative"
        remaining_tracks = [pool[i] for i in ut1]

        # --- stage 2: low-score dets vs still-unmatched *active* tracks (occlusion recovery).
        # Following ByteTrack, lost tracks are NOT re-associated here, a low-confidence box
        # is too weak to revive a long-gone identity, and doing so both over-trusts noise and
        # (since update() resets time_since_update) would leave un-cullable zombie tracks.
        stage2_pool = [t for t in remaining_tracks if t.state != "lost"]
        lost_unmatched = [t for t in remaining_tracks if t.state == "lost"]
        m2, ut2, ud2 = self._match(stage2_pool, low)
        for ti, di in m2:
            stage2_pool[ti].update(low[di])
        still_unmatched = [stage2_pool[i] for i in ut2] + lost_unmatched

        # --- unmatched high dets -> new tracks ---
        for di in ud1:
            self._spawn(high[di])

        # --- age out / mark lost ---
        for t in still_unmatched:
            if t.time_since_update > self.max_age:
                t.state = "dead"
            elif t.state == "tentative":
                t.state = "dead"          # tentative that missed -> discard
            else:
                t.state = "lost"
        self.tracks = [t for t in self.tracks if t.state != "dead"]

        for t in self.tracks:
            if t.time_since_update == 0:
                t.history.append((frame, t.bbox))
        return [t for t in self.tracks if t.state == "confirmed" and t.time_since_update == 0]

    def _spawn(self, det: Detection):
        kf = BBoxKalman()
        mean, cov = kf.initiate(_xyah(det.bbox))
        t = Track(track_id=self._next_id, cls=det.cls, mean=mean, cov=cov, kf=kf,
                  score=det.score, keypoints=det.keypoints, last_gt_id=det.gt_id)
        self._next_id += 1
        self.tracks.append(t)
        return t
