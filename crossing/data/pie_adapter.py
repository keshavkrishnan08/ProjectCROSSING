"""Adapter from real PIE / JAAD footage into the CROSSING schema.

The simulator lets us develop and *evaluate* against ground truth today; this adapter is
the path to real data. PIE (Pedestrian Intention Estimation) ships 6+ h of 1920x1080 30 fps
Toronto driving video with 1,842 pedestrian tracks, 738,970 pedestrian boxes, 2.35 M
traffic-object boxes, behaviour + crossing-intention labels, infrastructure annotations,
and synchronised ego speed/GPS/heading. JAAD is a complementary benchmark.

Because PIE is *ego-view* (a moving dashcam) rather than a fixed camera, the image->ground
homography changes per frame; :meth:`PIEAdapter.homography_for_frame` exposes that hook
(estimated from the road plane / vanishing point, or from ego motion + a flat-ground
assumption). Everything downstream — tracking, world model, intention, interaction, risk —
is identical to the simulated path, because it all speaks the same :mod:`crossing.schema`.

Expected on-disk layout (the official PIE devkit's):
    data/PIE/annotations/<set>/<video>_annt.xml           # boxes, behaviour, attributes
    data/PIE/annotations_attributes/.../<video>_attributes.xml
    data/PIE/PIE_clips/<set>/<video>.mp4

This module parses those into per-frame :class:`Detection` streams (with ``gt_id`` set to
the annotated track id, so the same evaluation harness applies) and a :class:`Scene`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import IMAGE_W, IMAGE_H, DATA_DIR
from ..schema import Detection, BBox, Scene
from ..worldmodel.homography import Homography, synthetic_camera_homography

# PIE behaviour/action label -> CROSSING canonical class (all pedestrians here)
_PIE_TO_CLS = {"pedestrian": "person", "ped": "person"}


class PIEAdapter:
    """Load a single PIE clip's annotations into detections + a scene.

    Only depends on the standard library for parsing; video decode (for a real detector
    instead of replaying boxes) uses the optional OpenCV path in
    :class:`crossing.perception.detector.YoloDetector`.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root or (DATA_DIR / "PIE"))

    # -- discovery --------------------------------------------------------------------
    def list_clips(self) -> list[str]:
        ann = self.root / "annotations"
        if not ann.exists():
            return []
        return sorted(str(p) for p in ann.rglob("*_annt.xml"))

    # -- parsing ----------------------------------------------------------------------
    def load_clip(self, annotation_xml: str):
        """Parse one ``*_annt.xml`` into (detections_per_frame, n_frames, scene).

        The PIE XML nests ``track`` elements (one per object) whose ``box`` children carry
        ``frame`` and ``xtl,ytl,xbr,ybr`` plus behaviour attributes. We emit a
        :class:`Detection` per box, tagging ``gt_id`` with the track id so evaluation works.
        """
        path = Path(annotation_xml)
        if not path.exists():
            raise FileNotFoundError(f"annotation not found: {path} (place PIE under {self.root})")
        tree = ET.parse(path)
        root = tree.getroot()

        by_frame: dict[int, list[Detection]] = {}
        max_frame = 0
        for track in root.iter("track"):
            label = (track.get("label") or "pedestrian").lower()
            cls = _PIE_TO_CLS.get(label, "person" if "ped" in label else label)
            tid = _stable_id(track.get("id") or track.get("obj_id") or label)
            for box in track.iter("box"):
                fr = int(box.get("frame"))
                max_frame = max(max_frame, fr)
                bb = BBox(float(box.get("xtl")), float(box.get("ytl")),
                          float(box.get("xbr")), float(box.get("ybr")))
                # occlusion attribute -> score proxy
                occ = box.get("occlusion")
                score = 0.9 if (occ in (None, "0", "none")) else 0.5
                by_frame.setdefault(fr, []).append(
                    Detection(frame=fr, cls=cls, bbox=bb, score=score, gt_id=tid))

        n_frames = max_frame + 1
        detections = [by_frame.get(f, []) for f in range(n_frames)]
        scene = self._default_scene(path.stem)
        return detections, n_frames, scene

    def _default_scene(self, name: str) -> Scene:
        """A best-effort scene for a dashcam clip: a forward-looking ground homography.

        For rigorous work, calibrate per clip from lane markings / vanishing point; here we
        provide a reasonable forward-road plane so the world model produces sane metres.
        """
        H = synthetic_camera_homography(IMAGE_W, IMAGE_H)
        return Scene(name=f"PIE:{name}", homography=H.to_list(),
                     image_w=IMAGE_W, image_h=IMAGE_H,
                     ground_bounds=[-2.0, 0.0, 42.0, 46.0], roads=[], crosswalks=[],
                     traffic_lights=[])

    def homography_for_frame(self, frame: int, ego=None) -> Homography:
        """Hook for per-frame (ego-motion-dependent) homography on dashcam data.

        With a calibrated intrinsic + a flat-ground assumption, ego yaw/pitch per frame
        yields a time-varying image->ground map. Without calibration we return the static
        forward-plane homography.
        """
        return Homography.from_list(self._default_scene("frame").homography)


def _stable_id(s: str) -> int:
    """Map an arbitrary PIE track id string to a stable positive integer."""
    try:
        return int("".join(ch for ch in s if ch.isdigit()) or "0") + 20000
    except ValueError:
        return (abs(hash(s)) % 90000) + 20000
