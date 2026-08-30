"""Face and licence-plate redaction.

We already localise every person and vehicle, so redaction is targeted rather than a blunt
whole-frame blur: the head region of each person box and the plate region of each vehicle
box are pixelated. Pixelation (mosaic) is used rather than a light Gaussian because it is
irreversible at this block size, which is the property that matters for de-identification.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RedactionPolicy:
    face_band: float = 0.34     # top fraction of a person box treated as head/face
    plate_band_top: float = 0.55  # plate sits in the lower-middle of a vehicle rear/front
    plate_band_bot: float = 0.88
    plate_inset: float = 0.18   # horizontal inset so we mosaic the central plate strip
    mosaic: int = 10            # target blocks across the redacted region
    enabled: bool = True

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "redacts": ["pedestrian faces", "vehicle licence plates"],
            "method": "targeted mosaic (irreversible at block size)",
            "applied": "in-pipeline, before any frame is written or served",
        }


REDACTION_POLICY = RedactionPolicy()


def _mosaic(frame, x1, y1, x2, y2, blocks):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return
    roi = frame[y1:y2, x1:x2]
    bw = max(1, (x2 - x1) // max(1, blocks))
    bh = max(1, (y2 - y1) // max(1, blocks))
    small = cv2.resize(roi, (max(1, (x2 - x1) // bw), max(1, (y2 - y1) // bh)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def redact_frame(frame, peds, vehicles, policy: RedactionPolicy = REDACTION_POLICY):
    """De-identify a frame in place: mosaic pedestrian heads and vehicle plate regions.

    ``peds`` / ``vehicles`` are dicts with a ``box`` of ``[x1, y1, x2, y2]`` in frame pixels
    (the same structures the safety pipeline already builds). Returns the frame for chaining.
    """
    if not policy.enabled:
        return frame
    for p in peds:
        x1, y1, x2, y2 = p["box"]
        _mosaic(frame, x1, y1, x2, y1 + (y2 - y1) * policy.face_band, policy.mosaic)
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        w = x2 - x1
        _mosaic(frame,
                x1 + w * policy.plate_inset, y1 + (y2 - y1) * policy.plate_band_top,
                x2 - w * policy.plate_inset, y1 + (y2 - y1) * policy.plate_band_bot,
                max(4, policy.mosaic // 2))
    return frame
