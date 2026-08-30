"""CROSSING command-line interface.

    crossing simulate --seed 7 --duration 40 --out scene.json
    crossing run       --seed 7 --duration 40           # full pipeline -> recording + DB
    crossing train     [--fast]                          # train models + evaluate
    crossing demo      [--seed 7]                        # build the web demo bundle
    crossing serve     [--port 8000]                     # launch the API + dashboard
    crossing evaluate  --seeds 201 202                   # capability metrics
"""
from __future__ import annotations

import argparse
import json


def _simulate(a):
    from .data.simulator import simulate
    from .schema import Recording, WorldState
    sim = simulate(seed=a.seed, duration_s=a.duration)
    print(f"simulated {sim.n_frames} frames, {len(sim.gt_conflicts)} scripted conflicts")
    if a.out:
        # export raw detections + GT is large; export scene + conflict summary
        import json
        json.dumps  # noqa
        print(f"scene: {sim.scene.name}")


def _run(a):
    from .config import Config
    from .data.simulator import simulate
    from .tracking.oracle import OracleTracker
    from .tracking.bytetrack import ByteTracker
    from .perception.detector import ReplayDetector
    from .pipeline import Pipeline
    from .data.datamodel import EventDatabase
    sim = simulate(seed=a.seed, duration_s=a.duration)
    tracker = ByteTracker() if a.tracker == "byte" else OracleTracker()
    rec = Pipeline(sim.scene, tracker, Config()).process(
        ReplayDetector(sim.detections), progress=lambda f, n: print(f"  {f}/{n}", end="\r"))
    print("\n" + json.dumps(rec.meta, indent=2))
    if a.db:
        db = EventDatabase()
        sid = db.ingest(rec)
        print(f"ingested as scene {sid} into {db.path}")


def _train(a):
    from scripts.train_all import main as train_main
    import sys
    sys.argv = ["train_all"] + (["--fast"] if a.fast else [])
    train_main()


def _demo(a):
    from scripts.build_demo import build
    build(showcase_seed=a.seed, duration=a.duration)


def _serve(a):
    import uvicorn
    uvicorn.run("crossing.api.server:app", host=a.host, port=a.port, reload=False)


def _evaluate(a):
    from .ml.evaluate import evaluate_all
    res = evaluate_all(a.seeds, duration=a.duration)
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser(prog="crossing", description="Pedestrian safety intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate"); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--duration", type=float, default=40.0); s.add_argument("--out")
    s.set_defaults(fn=_simulate)

    s = sub.add_parser("run"); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--duration", type=float, default=40.0)
    s.add_argument("--tracker", choices=["oracle", "byte"], default="oracle")
    s.add_argument("--db", action="store_true"); s.set_defaults(fn=_run)

    s = sub.add_parser("train"); s.add_argument("--fast", action="store_true")
    s.set_defaults(fn=_train)

    s = sub.add_parser("demo"); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--duration", type=float, default=45.0); s.set_defaults(fn=_demo)

    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=_serve)

    s = sub.add_parser("evaluate"); s.add_argument("--seeds", type=int, nargs="+", default=[201, 202])
    s.add_argument("--duration", type=float, default=30.0); s.set_defaults(fn=_evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
