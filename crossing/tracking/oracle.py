"""Assisted (oracle) tracker for annotated / simulated data.

Multi-object tracking on crowded, heavily-occluded street scenes fragments — that is a
real property of the problem, and CROSSING measures it honestly with :class:`ByteTracker`
in the evaluation suite (MOTA / IDF1 / fragmentation).

For *downstream analysis* on data that carries identity annotations (the simulator's
``gt_id``, or PIE/JAAD track ids), we don't want fragmentation to corrupt intention ramps
and interaction histories. The assisted tracker uses the annotated association to keep
identities persistent, while still:

* running every box through the same :class:`BBoxKalman` smoothing as the real tracker,
* honouring detection gaps/occlusion (a missing detection = a coasting, un-updated track),
* dropping false positives (``gt_id is None``).

So trajectories still look like tracker output — smoothed, with realistic gaps — just
without spurious ID switches. This is the mode the live demo and the risk analytics use;
it mirrors deploying a strong tracker or working from labelled footage.
"""
from __future__ import annotations

from ..schema import Detection
from .bytetrack import Track, _xyah
from .kalman import BBoxKalman


class OracleTracker:
    """Identity-stable tracker driven by annotated ``gt_id``."""

    def __init__(self, max_age: int = 45):
        self.max_age = max_age
        self.by_gt: dict[int, Track] = {}
        self._next_id = 1
        self.frame = -1

    def update(self, detections: list[Detection], frame: int) -> list[Track]:
        self.frame = frame
        for t in self.by_gt.values():
            t.predict()

        seen = set()
        for d in detections:
            if d.gt_id is None or d.cls == "traffic_light":
                continue                      # skip false positives / infra
            seen.add(d.gt_id)
            t = self.by_gt.get(d.gt_id)
            if t is None:
                kf = BBoxKalman()
                mean, cov = kf.initiate(_xyah(d.bbox))
                t = Track(track_id=self._next_id, cls=d.cls, mean=mean, cov=cov, kf=kf,
                          score=d.score, keypoints=d.keypoints, last_gt_id=d.gt_id)
                self._next_id += 1
                self.by_gt[d.gt_id] = t
            else:
                t.update(d)
            t.state = "confirmed"
            t.history.append((frame, t.bbox))

        # retire long-unseen ids
        dead = [g for g, t in self.by_gt.items() if t.time_since_update > self.max_age]
        for g in dead:
            del self.by_gt[g]

        return [t for t in self.by_gt.values() if t.last_gt_id in seen]
