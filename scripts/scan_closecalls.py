#!/usr/bin/env python3
"""Scan candidate clips for genuine close calls: sample several windows per video and report
the window with the highest peak hit-probability. Prints a ranked list so the best close-call
windows can be promoted onto the Live tab."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from scripts.safety_video import process

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# unexplored candidates worth scanning for close calls
CANDS = [
    "cross_a.mp4", "cross_b.mp4", "c_29093578.mp4", "c_34952428.mp4", "c_36673526.mp4",
    "cand_2103099.mp4", "cand_3121459.mp4", "yt_wMHb5u9_zfQ.mp4", "crossing_street.mp4",
    "street2.mp4", "people.mp4",
]

WIN = 45          # frames per probe window
N_STARTS = 6      # evenly spaced probes across the clip


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def scan(name):
    src = ASSETS / name
    if not src.exists():
        return None
    total = frame_count(src)
    if total < WIN + 10:
        starts = [max(0, total // 2 - WIN // 2)]
    else:
        lo, hi = int(total * 0.05), int(total * 0.92) - WIN
        step = max(1, (hi - lo) // (N_STARTS - 1))
        starts = list(range(lo, hi + 1, step))[:N_STARTS]
    best = {"peak": -1}
    for s in starts:
        try:
            meta = process(str(src), start=s, count=WIN, out_w=None, out_mp4=None)
        except Exception as e:
            print(f"    {name}@{s} err {e}", flush=True)
            continue
        fr = meta["frames"]
        peak = max((f["max_risk"] for f in fr), default=0.0)
        hot = sum(1 for f in fr if f["max_risk"] >= 0.78)
        if peak > best["peak"]:
            best = {"name": name, "start": s, "peak": round(peak, 3), "hot_frames": hot,
                    "total": total}
        print(f"    {name}@{s}: peak {round(peak*100)}%  hot {hot}", flush=True)
    return best


def main():
    names = sys.argv[1:] or CANDS
    results = []
    for name in names:
        print(f"[scan] {name}", flush=True)
        b = scan(name)
        if b and b["peak"] >= 0:
            results.append(b)
    results.sort(key=lambda r: -r["peak"])
    print("\n=== RANKED CLOSE CALLS ===")
    for r in results:
        print(f"  {r['peak']*100:3.0f}%  {r['name']:22s} start={r['start']:5d} hot={r['hot_frames']:2d} total={r['total']}")


if __name__ == "__main__":
    main()
