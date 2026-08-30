#!/usr/bin/env python3
"""Dense near-miss scan: slide a short window across each clip and report the windows with the
most sustained danger (frames above a high risk threshold), so genuine car-almost-hits-someone
moments float to the top rather than one-frame spikes."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from scripts.safety_video import process

ASSETS = Path(__file__).resolve().parent.parent / "assets"
WIN = 45
STRIDE = 60          # dense: a probe every ~2 seconds
HOT = 0.80           # a frame counts as "near miss" above this risk


def frame_count(p):
    cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release(); return n


def scan(name):
    src = ASSETS / name
    if not src.exists():
        print("missing", name); return []
    total = frame_count(src)
    out = []
    s = int(total * 0.03)
    hi = int(total * 0.95) - WIN
    while s <= hi:
        try:
            meta = process(str(src), start=s, count=WIN, out_w=None, out_mp4=None)
        except Exception as e:
            print(f"  {name}@{s} err {e}", flush=True); s += STRIDE; continue
        fr = meta["frames"]
        peak = max((f["max_risk"] for f in fr), default=0.0)
        hot = sum(1 for f in fr if f["max_risk"] >= HOT)
        if hot >= 3:
            out.append({"name": name, "start": s, "peak": round(peak, 3), "hot": hot})
            print(f"  * {name}@{s}: peak {round(peak*100)}% hot {hot}", flush=True)
        s += STRIDE
    return out


def main():
    names = sys.argv[1:] or ["cand_2103099.mp4", "cand_3121459.mp4", "c_29093578.mp4",
                             "yt_wMHb5u9_zfQ.mp4", "night.mp4"]
    res = []
    for n in names:
        print(f"[dense] {n}", flush=True)
        res.extend(scan(n))
    res.sort(key=lambda r: (-r["hot"], -r["peak"]))
    print("\n=== TOP SUSTAINED NEAR-MISSES (by hot frames) ===")
    for r in res[:25]:
        print(f"  hot {r['hot']:2d}  peak {r['peak']*100:3.0f}%  {r['name']}@{r['start']}")


if __name__ == "__main__":
    main()
