"""Perception — the sensor layer.

The object detector is *just the sensor*; the intelligence is everything downstream. This
module defines a small :class:`Detector` interface so the rest of CROSSING is agnostic to
where boxes come from:

* :class:`ReplayDetector` — serves pre-computed detections (the simulator's rendered
  boxes, or imported PIE/JAAD annotations). Used everywhere by default.
* :class:`YoloDetector` — runs a real Ultralytics YOLO model on a video, mapping COCO
  classes into CROSSING's taxonomy. Auto-used when ``ultralytics`` + a video are present;
  otherwise the platform runs perfectly on replayed detections.

Swapping the sensor changes nothing downstream — that is the whole point.
"""
from __future__ import annotations

from typing import Iterable

from ..config import CLASS_ALIASES
from ..schema import Detection, BBox


class Detector:
    """Interface: yield the detections for a given frame index."""

    def detect(self, frame_idx: int) -> list[Detection]:
        raise NotImplementedError

    @property
    def n_frames(self) -> int:
        raise NotImplementedError


class ReplayDetector(Detector):
    """Serves detections already computed per frame (simulator / annotations)."""

    def __init__(self, detections_per_frame: list[list[Detection]]):
        self._dets = detections_per_frame

    def detect(self, frame_idx: int) -> list[Detection]:
        if 0 <= frame_idx < len(self._dets):
            return self._dets[frame_idx]
        return []

    @property
    def n_frames(self) -> int:
        return len(self._dets)


class YoloDetector(Detector):
    """Real YOLO detection over a video file (optional; needs ultralytics + opencv).

    Detections are decoded lazily and cached. Falls back with a clear message if the
    optional dependencies are missing so the platform never hard-fails.
    """

    def __init__(self, video_path: str, model_name: str = "yolov8n.pt",
                 conf: float = 0.2, classes: Iterable[str] | None = None):
        try:
            from ultralytics import YOLO  # noqa
            import cv2  # noqa
        except Exception as e:  # pragma: no cover - optional path
            raise RuntimeError(
                "YoloDetector requires `ultralytics` and `opencv-python`. "
                "Install the [accel] extra, or use ReplayDetector with simulated/annotated "
                f"data. (import error: {e})"
            )
        from ultralytics import YOLO
        import cv2
        self._cv2 = cv2
        self.model = YOLO(model_name)
        self.conf = conf
        self.cap = cv2.VideoCapture(video_path)
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cache: dict[int, list[Detection]] = {}

    @property
    def n_frames(self) -> int:
        return self._n

    def detect(self, frame_idx: int) -> list[Detection]:  # pragma: no cover - optional path
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return []
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        dets = []
        names = res.names
        for b in res.boxes:
            raw = names[int(b.cls)]
            cls = CLASS_ALIASES.get(raw)
            if cls is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append(Detection(frame=frame_idx, cls=cls, bbox=BBox(x1, y1, x2, y2),
                                  score=float(b.conf)))
        self._cache[frame_idx] = dets
        return dets
