#!/usr/bin/env python3
"""Run REAL object detection (YOLOv3, ONNX) on a REAL video and draw the boxes.

No torch / ultralytics required — uses onnxruntime + opencv, both of which install on
Python 3.14. Produces a boxed .mp4 and (optionally) a per-frame detection JSON so a web
page can overlay the boxes live.

Preprocessing note: the ONNX Model-Zoo YOLOv3 expects a plain 416x416 resize (it undoes
the aspect distortion internally using the ``image_shape`` input) — NOT a letterbox.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ASSETS = Path(__file__).resolve().parent.parent / "assets"
COCO = (ASSETS / "coco.names").read_text().strip().split("\n")

# semantic colours (BGR): green=pedestrian, blue=vehicle, amber=signal, purple=two-wheeler
_PERSON = (120, 210, 90)
_VEHICLE = (255, 162, 90)
_SIGNAL = (82, 154, 255)
_TWO_WHEEL = (234, 146, 199)
_VEHICLES = {"car", "truck", "bus", "van"}
_TWO = {"bicycle", "motorbike", "motorcycle"}


def color_for(cls_name: str):
    if cls_name == "person":
        return _PERSON
    if cls_name in _VEHICLES:
        return _VEHICLE
    if cls_name in _TWO:
        return _TWO_WHEEL
    if cls_name == "traffic light":
        return _SIGNAL
    return (150, 200, 120)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def cross_class_nms(dets, thresh=0.6):
    """Drop duplicate boxes for the same object detected as two classes (keep higher conf)."""
    order = sorted(dets, key=lambda d: -d["conf"])
    kept = []
    for d in order:
        if all(_iou(d["box"], k["box"]) < thresh for k in kept):
            kept.append(d)
    return kept


class YoloOnnx:
    def __init__(self, model=ASSETS / "yolov3.onnx", size=416, conf=0.0):
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.size = size
        self.conf = conf

    def detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (self.size, self.size)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]
        boxes, scores, indices = self.sess.run(
            None, {"input_1": inp, "image_shape": np.array([[H, W]], np.float32)})
        rows = indices if indices.ndim == 2 else indices[0]
        out = []
        for i in rows:
            c, bi = int(i[1]), int(i[2])
            y1, x1, y2, x2 = [float(v) for v in boxes[0][bi]]
            sc = float(scores[0][c][bi])
            if sc >= self.conf:
                out.append({"cls": COCO[c], "cls_id": c, "conf": round(sc, 3),
                            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
        return out


def draw(frame, dets, thickness=2):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        col = color_for(d["cls"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)
        label = f'{d["cls"]} {d["conf"]*100:.0f}%'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(0, y1 - th - 7)
        cv2.rectangle(frame, (x1, ytop), (x1 + tw + 8, ytop + th + 7), col, -1)
        cv2.putText(frame, label, (x1 + 4, ytop + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    return frame


def process(video, start=0, count=240, keep=None, out_w=None, out_mp4=None,
            frames_dir=None, jpeg_q=62):
    yolo = YoloOnnx()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer, per_frame, n = None, [], 0
    if frames_dir:
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
    for k in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        dets = yolo.detect(frame)
        if keep is not None:
            dets = [d for d in dets if d["cls"] in keep]
        dets = cross_class_nms(dets)
        draw(frame, dets)
        if out_w:
            h = int(frame.shape[0] * out_w / frame.shape[1])
            frame = cv2.resize(frame, (out_w, h))
        if out_mp4:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(out_mp4), fourcc, fps,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if frames_dir:
            cv2.imwrite(str(Path(frames_dir) / f"f{n:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        per_frame.append({"frame": start + k, "dets": dets})
        n += 1
        if n % 25 == 0:
            print(f"  processed {n}/{count} frames", flush=True)
    if writer:
        writer.release()
    return {"fps": fps, "n_frames": n, "frames": per_frame}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ASSETS / "vtest.avi"))
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--count", type=int, default=240)
    ap.add_argument("--out-w", type=int, default=720)
    ap.add_argument("--mp4", default=str(ASSETS / "detected.mp4"))
    ap.add_argument("--frames", default=str(ASSETS / "frames"))
    ap.add_argument("--json", default=str(ASSETS / "detections.json"))
    a = ap.parse_args()
    keep = {"person", "bicycle", "car", "motorbike", "motorcycle", "bus", "truck",
            "backpack", "handbag", "traffic light"}
    meta = process(a.video, a.start, a.count, keep=keep, out_w=a.out_w,
                   out_mp4=a.mp4, frames_dir=a.frames)
    Path(a.json).write_text(json.dumps(meta))
    print(f"done: {meta['n_frames']} frames @ {meta['fps']} fps -> {a.mp4}")
