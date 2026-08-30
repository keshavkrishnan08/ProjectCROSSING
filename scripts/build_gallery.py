#!/usr/bin/env python3
"""Process several DISTINCT real environments through the safety pipeline: four full clips
for the Live tab and one risk-scored still from each environment for the gallery."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from scripts.safety_video import process
from crossing.streaming import build_fleet
from crossing.monitoring import temporal_drift_report
from crossing.privacy import REDACTION_POLICY
from crossing.provenance import MODEL_VERSION

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MEDIA = ROOT / "frontend" / "media"
FRONTEND = ROOT / "frontend"
MEDIA.mkdir(parents=True, exist_ok=True)

# (asset, start, count, scene, environment description [no commas]) ordered.
# The first LIVE_N are the close-call clips shown on the Live tab; the rest are calm scenes
# processed only so the "collision risk is rare" histogram reflects a representative corpus.
LIVE_N = 6
# The Live windows are trimmed tight around each clip's car-almost-hits moment so the near
# miss is what plays, not a long calm lead-in (the peak sits ~0.8 s into each window).
ENVS = [
    ("yt_mE44pLMpfkw.mp4", 1726, 58, "Mid-block crossing", "a pedestrian crosses in front of a moving car"),
    ("crossing_street.mp4", 399, 56, "Dusk city crossing", "a car closes on a pedestrian at dusk"),
    ("c_36306846.mp4", 244, 60, "Downtown crosswalk", "a rider bears down on people at a crosswalk"),
    ("c_14365420.mp4", 104, 56, "Street corner", "a car swings past people on a corner"),
    ("cross_a.mp4", 21, 58, "Crossing signal", "a pedestrian steps off as a car crosses"),
    ("c_34952428.mp4", 285, 56, "Marked crosswalk", "a truck rolls through a crowded crosswalk"),
    ("c_20221532.mp4", 300, 250, "Busy avenue", "a wide avenue thick with cars and people"),
    ("c_3552510.mp4", 60, 70, "Pedestrian street", "a lively pedestrian street"),
    ("c_3894943.mp4", 20, 70, "Shared street", "a narrow shared street of walkers and cars"),
    ("vtest.avi", 40, 70, "Open plaza", "an open plaza with foot traffic and a parked van"),
]


def peak_frame(meta):
    fr = meta["frames"]
    return max(range(len(fr)), key=lambda k: fr[k]["max_risk"]), fr


def main():
    gallery, fleet_scenes, lineage_records, risk_streams = [], [], [], {}
    all_risk = []                                      # every frame's peak risk, for the histogram
    for i, (asset, start, count, scene, env) in enumerate(ENVS, start=1):
        src = ASSETS / asset
        if not src.exists():
            print("skip missing", asset); continue
        mp4 = ASSETS / f"env{i}.mp4"
        print(f"[{i}] {scene}: processing {count} frames of {asset}...", flush=True)
        meta = process(str(src), start, count, out_w=960, out_mp4=str(mp4), json_out=None)
        risk_series = [round(f["max_risk"], 3) for f in meta["frames"]]
        risk_streams[scene] = risk_series
        all_risk.extend(round(f["max_risk"] * 100) for f in meta["frames"])
        peak_pct = round(max(risk_series, default=0.0) * 100)
        at_risk_now = max((f["at_risk"] for f in meta["frames"]), default=0)
        # Live videos: the first LIVE_N close-call clips -> served as safety1..N with counts + lineage
        if i <= LIVE_N:
            shutil.copy(mp4, MEDIA / f"safety{i}.mp4")
            counts = [[f["n_ped"], f["at_risk"], round(f["max_risk"] * 100)] for f in meta["frames"]]
            json.dump({"fps": meta["fps"], "frames": counts, "lineage": meta.get("lineage")},
                      open(MEDIA / f"safety{i}_counts.json", "w"))
            if meta.get("lineage"):
                lineage_records.append({"scene": scene, "env": env, "video": f"/media/safety{i}.mp4",
                                        **meta["lineage"]})
        # one gallery still from this environment: its peak-risk frame
        pk, fr = peak_frame(meta)
        cap = cv2.VideoCapture(str(mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, pk); ok, frame = cap.read()
        if ok:
            w = 620; h = int(frame.shape[0] * w / frame.shape[1])
            cv2.imwrite(str(MEDIA / f"env{i}.jpg"), cv2.resize(frame, (w, h)),
                        [cv2.IMWRITE_JPEG_QUALITY, 84])
            gallery.append({"img": f"/media/env{i}.jpg", "scene": scene, "env": env,
                            "risk": round(fr[pk]["max_risk"] * 100), "ped": fr[pk]["n_ped"]})
        fleet_scenes.append({"name": scene, "peak_risk": peak_pct, "at_risk": at_risk_now,
                             "alerts": 1 if peak_pct >= 78 else 0})
        print(f"    peak risk {peak_pct}%  ({fr[pk]['n_ped']} people)", flush=True)
    json.dump(gallery, open(MEDIA / "gallery.json", "w"))

    # ---- governance artifacts (fleet scale, provenance, live monitoring) ----
    fleet = build_fleet(fleet_scenes)
    fs = fleet.status()
    fs["redaction"] = REDACTION_POLICY.summary()
    fs["model_version"] = MODEL_VERSION
    json.dump(fs, open(MEDIA / "fleet.json", "w"))

    json.dump({"model_version": MODEL_VERSION, "records": lineage_records},
              open(MEDIA / "lineage.json", "w"))

    drift = temporal_drift_report(risk_streams, bins=10)   # per-feed shift over time
    reliab = json.load(open(ROOT / "artifacts" / "metrics" / "risk.json")).get("reliability", {})
    json.dump({"model_version": MODEL_VERSION, "drift": drift, "calibration": reliab},
              open(MEDIA / "monitoring.json", "w"))

    # ---- histogram over the whole corpus (close-call + calm) for the "risk is rare" chart ----
    bins = [0] * 10
    for r in all_risk:
        bins[min(9, int(r) // 10)] += 1
    json.dump({"bins": bins, "frames": len(all_risk)}, open(MEDIA / "risk_hist.json", "w"))
    print("live:", LIVE_N, "| gallery entries:", len(gallery), "| corpus frames:", len(all_risk),
          "| hist bins:", bins)


if __name__ == "__main__":
    main()
