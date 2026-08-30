#!/usr/bin/env python3
"""Train every learned model and run the full evaluation suite, writing metrics to disk.

Usage:
    python -m scripts.train_all [--duration 30] [--train N] [--fast]

Splits are strictly by scene (leakage-free). Outputs land in ``artifacts/metrics/``:
    risk.json, intention.json, capabilities.json, summary.json
and models in ``artifacts/models/``.
"""
from __future__ import annotations

import argparse
import json
import time

from crossing.config import METRICS_DIR
from crossing.ml.train_risk import train_risk
from crossing.ml.train_intention import train_intention
from crossing.ml.evaluate import evaluate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--train", type=int, default=8, help="number of training scenes")
    ap.add_argument("--fast", action="store_true", help="fewer scenes / epochs for a quick run")
    args = ap.parse_args()

    n_train = 4 if args.fast else args.train
    dur = 20.0 if args.fast else args.duration
    train_seeds = list(range(1, 1 + n_train))
    val_seeds = [101, 102, 103][: (1 if args.fast else 3)]
    test_seeds = [201, 202, 203, 204][: (2 if args.fast else 4)]
    epochs = 20 if args.fast else 45

    print(f"== CROSSING training ==  train={train_seeds} val={val_seeds} test={test_seeds} dur={dur}s")
    t0 = time.time()

    print("\n--- Risk model + baseline ladder ---")
    risk = train_risk(train_seeds, val_seeds, test_seeds, duration=dur, save=True)
    (METRICS_DIR / "risk.json").write_text(json.dumps(risk, indent=2))

    print("\n--- Crossing-intention GRU ---")
    intent = train_intention(train_seeds, val_seeds, test_seeds, duration=dur,
                             epochs=epochs, save=True)
    (METRICS_DIR / "intention.json").write_text(json.dumps(intent, indent=2))

    print("\n--- Capability evaluation (detection / tracking / trajectory / intention) ---")
    caps = evaluate_all(test_seeds, duration=dur, use_learned=True)
    (METRICS_DIR / "capabilities.json").write_text(json.dumps(caps, indent=2))

    # headline summary
    lr = risk["table"]["learned"]; base = risk["table"]["ttc+motion"]
    summary = {
        "risk_auroc_learned": lr["auroc"],
        "risk_auroc_ttc_motion": base["auroc"],
        "risk_auprc_learned": lr["auprc"],
        "risk_ece_learned": lr["ece"],
        "risk_ece_physics": risk["table"]["physics"]["ece"],
        "intention_auroc": intent["auroc"],
        "intention_lead_time_s": caps["intention"]["mean_lead_time_s"],
        "detection_f1": caps["detection"]["f1"],
        "tracking_mota": caps["tracking"]["mota"],
        "tracking_idf1": caps["tracking"]["idf1"],
        "trajectory_minADE_m": caps["trajectory"]["minADE_m"],
        "trajectory_minFDE_m": caps["trajectory"]["minFDE_m"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n== SUMMARY ==")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print(f"\nMetrics written to {METRICS_DIR}")


if __name__ == "__main__":
    main()
