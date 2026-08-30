#!/usr/bin/env python3
"""Process several DISTINCT real environments through the safety pipeline: four full clips
for the Live tab and one risk-scored still from each environment for the gallery."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from scripts.safety_video import process

ASSETS = Path(__file__).resolve().parent.parent / "assets"
MEDIA = Path(__file__).resolve().parent.parent / "frontend" / "media"
MEDIA.mkdir(parents=True, exist_ok=True)

# (asset, start, count, scene, environment description [no commas]) ordered; first 4 = Live videos
ENVS = [
    ("yt_mE44pLMpfkw.mp4", 1727, 78, "Mid-block crossing", "a pedestrian crosses right in front of a moving car"),
    ("traffic2.mp4", 180, 260, "Junction with a turning bus", "a daytime junction where a bus turns across the crossing"),
    ("c_20221532.mp4", 300, 250, "Busy avenue", "a wide avenue thick with cars and people"),
    ("c_36306846.mp4", 120, 250, "Downtown crosswalk", "a downtown crosswalk on a bright day"),
    ("c_14365420.mp4", 60, 70, "Street corner", "a busy street corner"),
    ("c_3552510.mp4", 60, 70, "Pedestrian street", "a lively pedestrian street"),
    ("c_3894943.mp4", 20, 70, "Shared street", "a narrow shared street of walkers and cars"),
    ("vtest.avi", 40, 70, "Open plaza", "an open plaza with foot traffic and a parked van"),
]


def peak_frame(meta):
    fr = meta["frames"]
    return max(range(len(fr)), key=lambda k: fr[k]["max_risk"]), fr


def main():
    gallery = []
    for i, (asset, start, count, scene, env) in enumerate(ENVS, start=1):
        src = ASSETS / asset
        if not src.exists():
            print("skip missing", asset); continue
        mp4 = ASSETS / f"env{i}.mp4"
        print(f"[{i}] {scene}: processing {count} frames of {asset}...", flush=True)
        meta = process(str(src), start, count, out_w=960, out_mp4=str(mp4), json_out=None)
        # Live videos: first four -> served as safety1..4 with per-frame counts
        if i <= 4:
            shutil.copy(mp4, MEDIA / f"safety{i}.mp4")
            counts = [[f["n_ped"], f["at_risk"], round(f["max_risk"] * 100)] for f in meta["frames"]]
            json.dump({"fps": meta["fps"], "frames": counts},
                      open(MEDIA / f"safety{i}_counts.json", "w"))
        # one gallery still from this environment: its peak-risk frame
        pk, fr = peak_frame(meta)
        cap = cv2.VideoCapture(str(mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, pk); ok, frame = cap.read()
        if ok:
            w = 620; h = int(frame.shape[0] * w / frame.shape[1])
            cv2.imwrite(str(MEDIA / f"env{i}.jpg"), cv2.resize(frame, (w, h)),
                        [cv2.IMWRITE_JPEG_QUALITY, 84])
            gallery.append({"img": f"/media/env{i}.jpg", "scene": scene, "env": env,
                            "risk": round(fr[pk]["max_risk"] * 100), "ped": fr[pk]["n_ped"]})
        print(f"    peak risk {round(fr[pk]['max_risk']*100)}%  ({fr[pk]['n_ped']} people)", flush=True)
    json.dump(gallery, open(MEDIA / "gallery.json", "w"))
    print("gallery entries:", len(gallery))


if __name__ == "__main__":
    main()
