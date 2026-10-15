#!/usr/bin/env python3
"""Pedestrian-safety intelligence on real video.

Beyond boxes: detect (YOLOv3) -> track (CROSSING's Kalman/ByteTrack) -> estimate each
pedestrian's **probability of being hit** from their tracked motion and every nearby
vehicle's speed and closing rate, then paint the box by that risk (green = safe, red =
danger) with a live percentage. Vehicles are drawn in one neutral colour. Two colours do
the work: a safe->danger gradient for people, one tone for vehicles.

Everything is the model's own output on real footage, the detections are YOLOv3, the
velocities come from the tracker's Kalman filter, and the hit-probability is computed from
the depth-aware image geometry frame by frame.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from scripts.detect_video import YoloOnnx, cross_class_nms
from crossing.schema import Detection, BBox
from crossing.tracking.bytetrack import ByteTracker
from crossing.privacy import redact_frame
from crossing.provenance import trace_risk
from crossing.risk.hit import foot_point, hit_probability

ASSETS = Path(__file__).resolve().parent.parent / "assets"
VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorbike", "motorcycle"}

# --- palette (BGR): a safe->danger gradient for people, one neutral tone for vehicles ---
_SAFE = np.array([120, 200, 90], float)      # green
_DANGER = np.array([60, 60, 235], float)     # red
_VEHICLE = np.array([210, 180, 140], float)  # muted slate-blue


def risk_color(r):
    c = _SAFE + (_DANGER - _SAFE) * float(np.clip(r, 0, 1))
    return tuple(int(v) for v in c)


def draw(frame, peds, vehicles):
    for v in vehicles:
        x1, y1, x2, y2 = [int(t) for t in v["box"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(int(c) for c in _VEHICLE), 2)
    for p in peds:
        x1, y1, x2, y2 = [int(t) for t in p["box"]]
        r = p["risk"]
        col = risk_color(r)
        th = 2 if r < 0.5 else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, th)
        label = f'{r*100:.0f}% risk'
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ytop = max(0, y1 - tht - 9)
        cv2.rectangle(frame, (x1, ytop), (x1 + tw + 10, ytop + tht + 9), col, -1)
        cv2.putText(frame, label, (x1 + 5, ytop + tht + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def process(video, start=0, count=250, out_w=960, out_mp4=None, json_out=None, redact=True):
    yolo = YoloOnnx()
    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.25, n_init=2, max_age=30)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer, per_frame, n = None, [], 0
    risk_ema = {}                                     # track_id -> smoothed risk
    history = {}                                      # track_id -> detection provenance
    pos_hist = {}                                     # track_id -> recent foot positions
    peak = {"risk": -1.0}                             # highest-risk instant, for lineage
    keep = {"person", "bicycle"} | VEHICLE_CLASSES
    for k in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        raw = [d for d in yolo.detect(frame) if d["cls"] in keep]
        raw = cross_class_nms(raw)
        dets = [Detection(frame=k, cls=d["cls"],
                          bbox=BBox(*d["box"]), score=d["conf"]) for d in raw]
        tracks = tracker.update(dets, k)
        peds, vehicles = [], []
        for t in tracks:
            b = t.bbox
            vx, vy = float(t.mean[4]), float(t.mean[5])
            item = {"id": t.track_id, "cls": t.cls, "box": [b.x1, b.y1, b.x2, b.y2],
                    "vx": vx, "vy": vy, "speed": round(math.hypot(vx, vy), 2)}
            (peds if t.cls in ("person", "bicycle") else vehicles).append(item)
            hp = history.setdefault(t.track_id, {"cls": t.cls, "conf": [], "first": k, "last": k})
            hp["conf"].append(round(float(t.score), 3)); hp["last"] = k
            dq = pos_hist.setdefault(t.track_id, deque(maxlen=10))
            dq.append(foot_point([b.x1, b.y1, b.x2, b.y2]))
        # Robust motion per vehicle: net foot displacement over recent frames, which cancels the
        # oscillatory box jitter a parked car shows. A vehicle counts as moving only if it has
        # actually travelled; a brand-new track (too little history) is treated as not moving so a
        # transient Kalman velocity spike cannot raise a false alarm.
        for v in vehicles:
            dq = pos_hist.get(v["id"])
            vh = max(v["box"][3] - v["box"][1], 12.0)
            if dq and len(dq) >= 3:
                (x0, y0), (x1, y1) = dq[0], dq[-1]
                dspeed = math.hypot(x1 - x0, y1 - y0) / (len(dq) - 1)
                v["dspeed"] = round(dspeed, 2)
                v["moving"] = dspeed > max(0.4, 0.012 * vh)
            else:
                v["dspeed"] = 0.0
                v["moving"] = False
        frame_w = frame.shape[1]
        for p in peds:
            r, detail = hit_probability(p, vehicles, fps=fps, frame_w=frame_w, explain=True)
            prev = risk_ema.get(p["id"], r)
            r = 0.6 * prev + 0.4 * r                  # temporal smoothing
            risk_ema[p["id"]] = r
            p["risk"] = round(r, 3)
            if r > peak["risk"]:                      # remember the derivation of the peak score
                peak = {"risk": r, "frame": k, "ped_id": p["id"], "detail": detail}
        # De-identify (faces + plates) BEFORE anything is written, then draw the safety overlay.
        if redact:
            redact_frame(frame, peds, vehicles)
        draw(frame, peds, vehicles)
        if out_w:
            h = int(frame.shape[0] * out_w / frame.shape[1])
            frame = cv2.resize(frame, (out_w, h))
        if out_mp4:
            if writer is None:
                writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"),
                                         fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        per_frame.append({
            "n_ped": len(peds), "n_veh": len(vehicles),
            "max_risk": round(max([p["risk"] for p in peds], default=0.0), 3),
            "at_risk": sum(1 for p in peds if p["risk"] >= 0.5),
        })
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{count}", flush=True)
    if writer:
        writer.release()
    # Lineage for the peak-risk instant: bind the score to its detections, track, and vehicle.
    lineage = None
    if peak.get("detail"):
        lineage = trace_risk(
            peak["ped_id"], peak["detail"].get("driver_id"), peak["risk"], peak["frame"],
            {k: peak["detail"][k] for k in ("d_real", "align", "prox", "path", "ttc_boost",
                                            "closing_ms", "veh_speed")},
            history, note="peak-risk instant of the clip",
        ).as_dict()
    meta = {"fps": fps, "n_frames": n, "frames": per_frame, "lineage": lineage,
            "redacted": bool(redact)}
    if json_out:
        Path(json_out).write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--out-w", type=int, default=960)
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    m = process(a.video, a.start, a.count, a.out_w, a.mp4, a.json)
    print(f"done: {m['n_frames']} frames @ {m['fps']} fps -> {a.mp4}")
