# CROSSING — pedestrian‑vehicle safety intelligence

> **A street isn't a collection of objects. It's a constantly changing network of
> interactions. CROSSING models that network in real time.**

CROSSING turns ordinary street video into a continuously‑updated model of every person,
vehicle, interaction, risk event, and emerging danger in an environment. The object
detector is *just the sensor*; the project is the intelligence layer on top of it:

```
video  →  perception (detect + pose)  →  tracking (persistent IDs)
       →  world model (image → ground metres; velocity / accel / heading / jerk)
       →  behaviour (crossing intention, attention)  →  prediction (multi‑modal trajectories)
       →  interaction engine (TTC / DCPA / TCPA / PET / path‑conflict)
       →  risk engine (calibrated) + near‑miss detection + explanation + counterfactuals
       →  event graph + city hotspots  →  operational dashboard
```

**Observe → Understand → Predict → Act.**

## Live demo

**→ https://crossing-pi.vercel.app** — real YOLOv3 detection on real street video from
four distinct environments, each pedestrian box painted by its depth‑aware
probability‑of‑being‑hit (green → red), with live commentary on every close call.

Run it locally with the full API (calibrated metrics, near‑miss feed, hotspots):

```bash
pip install -r requirements.txt
python -m uvicorn crossing.api.server:app --port 8000   # → http://localhost:8000
```

`/` is the safety page above; `/dashboard` is the full cinematic pipeline; `/api/metrics`
serves the measured model numbers.

---

## What it does (measured on held‑out test scenes)

| Capability | Metric | Result |
|---|---|---|
| Near‑miss **risk** prediction | AUROC (learned) vs. TTC+motion baseline | **0.951** vs. 0.868 |
| Risk **calibration** | Expected Calibration Error (learned vs. physics) | **0.002** vs. 0.368 |
| Crossing **intention** | AUROC · mean early‑warning before road entry | **0.961** · **2.8 s** |
| **Detection** | precision / recall / F1 | 0.999 / 0.909 / 0.952 |
| **Tracking** | MOTA / IDF1 | 0.745 / 0.702 |
| **Trajectory** | minADE / minFDE | 0.83 m / 1.27 m |

The headline research result: **a learned interaction model beats an engineered
TTC‑based risk score at predicting near‑misses before they happen, *and* is calibrated** —
"risk 80" really does mean ≈80% of such situations become near‑misses.

Every number is produced by `python -m scripts.train_all` and written to
`artifacts/metrics/`; nothing here is hand‑entered.

---

## Governance (production posture)

A safety model that watches public streets has to be operable and accountable, not just
accurate. Four layers make it deployable:

* **Provenance** (`crossing/provenance/`) — every risk score is content‑addressed to the
  model that made it and carries its full lineage: which detections, over how many frames,
  at what confidence, tracked into which actor, closing with which vehicle, through which
  metric components. The **Trace** tab renders that chain. "Where did this 97 come from?"
  always has an answer.
* **Privacy** (`crossing/privacy/`) — faces and licence plates are mosaic‑redacted *inside*
  the pipeline, before any frame is written or served, so raw imagery never leaves the
  processing step. The de‑identified frames are what the demo actually plays.
* **Access control + audit** (`crossing/security/`) — the API is role‑gated (`viewer` /
  `operator` / `analyst` / `admin`); the raw feed, the lineage, and the audit log each need
  a higher scope, and every governed read is written to an append‑only audit log.
* **Scale + monitoring** (`crossing/streaming/`, `crossing/monitoring/`) — a fleet registry
  models many camera feeds with health and backpressure (the **Fleet** tab), and drift
  (population‑stability + KS) plus calibration tracking watch whether the live input is still
  the distribution the model was built for (the **Monitor** tab).

Governed endpoints: `/api/fleet`, `/api/lineage`, `/api/monitoring`, `/api/audit`,
`/api/whoami`. Try them with `X-API-Key: crossing-analyst-key`.

---

## The interactive demo

A single self‑contained page (`frontend/index.html`) plays the whole story as a cinematic,
auto‑playing product tour over a real processed scene: perception → tracking → world model
→ crossing intent → multi‑modal prediction → **near‑miss freeze** → risk decomposition →
counterfactual ("what if the car were 5 mph slower?") → city‑scale hotspot map → the
measured results. Two synchronised views — a **camera** overlay and a **digital‑twin**
top‑down map in ground metres — carry it.

```bash
crossing serve            # then open http://127.0.0.1:8000
```

---

## Quickstart

```bash
pip install -r requirements.txt          # numpy/scipy/pandas/sklearn/networkx/fastapi
pip install -e .                          # installs the `crossing` CLI

crossing simulate --seed 7 --duration 40  # generate a scene (ground truth)
crossing run --seed 7 --db                # full pipeline → event database
crossing train                            # train models + full evaluation (~3 min)
crossing demo                             # build the web‑demo data bundle
crossing serve                            # launch the dashboard + JSON API
crossing evaluate --seeds 201 202         # capability metrics on unseen scenes
```

No GPU, no dataset download, and no heavyweight ML frameworks are required — the entire
platform runs on NumPy + scikit‑learn. `torch` / `ultralytics` / `xgboost` are **optional
accelerators**, auto‑detected when present.

---

## How it works

### Data — a physically‑grounded simulator, and a path to real footage
Developing and *evaluating* an intelligence stack needs ground truth. `crossing.data.simulator`
generates signalised‑intersection scenes with IDM car‑following, permissive turns, a
signal controller, gap‑acceptance pedestrians with observable **crossing‑intention
dynamics**, social‑force spacing, and **seeded conflict scenarios** (permissive‑left,
red‑light jaywalk, occluded emergence) that reliably produce genuine near‑misses. It
renders each actor to an image‑space detection through the scene homography with calibrated
noise, misses, false positives, and depth‑ordered occlusion — so tracking has real work to
do — and back‑references ground truth for evaluation. Every scene is deterministic in its
seed, so train/val/test splits are **by scene** (never by frame → no leakage).

Real **PIE** / **JAAD** footage plugs into the exact same schema through
`crossing.data.pie_adapter`; everything downstream is capture‑agnostic.

### Perception → tracking → world model
YOLO‑style detection (or replayed/annotated boxes) → a **ByteTrack**‑style tracker
(two‑stage high/low‑score association, Kalman motion, lost‑track buffer for occlusion) →
a homography that lifts each ground‑contact point onto the road plane, where a
constant‑acceleration Kalman filter yields stable position/velocity/acceleration/heading/
jerk in **metres**. Reconstruction error on unseen scenes: ~0.16 m (pedestrians) / 0.45 m
(vehicles).

### Behaviour, prediction, interaction, risk
* **Crossing intention** — a from‑scratch NumPy **GRU** (back‑prop‑through‑time,
  gradient‑checked) over 1.5 s of behavioural features, emitting a *calibrated probability
  that evolves over time* (0.21 → 0.68 → 0.94), never a boolean.
* **Prediction** — multiple goal‑conditioned futures per entity (continue / stop / turn
  back), weighted by intention and kinematics.
* **Interaction** — closed‑form conflict geometry (TTC, DCPA/TCPA, PET, path‑conflict),
  unit‑tested against hand‑computed cases.
* **Risk** — an interpretable physics fusion (which doubles as the *explanation* — each
  factor's share of the score) benchmarked against a **calibrated gradient‑boosted** model.
  Near‑miss labels use the *realised* closest separation, not noise‑biased predicted CPA.
* **Counterfactuals** — replay the two actors' recorded trajectories under a modified
  control (slower vehicle / delayed pedestrian) and recompute the geometry. Explicitly a
  model simulation, not causal proof.

### Event graph + city intelligence
Per scene, a NetworkX knowledge graph of entities → events → zones. Across many scenes, a
`CityAnalyzer` builds a spatial risk map, ranked hotspots, and systemic patterns ("where is
the environment systematically unsafe, and why?"). Everything persists to a SQLite event
database you can query in plain SQL.

---

## Layout

```
crossing/
  schema.py            unified data model (entities, interactions, events, recordings)
  config.py            coordinate conventions, physical constants, risk thresholds
  data/                simulator · PIE/JAAD adapter · SQLite event DB
  perception/          detector (YOLO/replay) · pose cues
  tracking/            Kalman filters · ByteTrack · assisted tracker
  worldmodel/          homography · motion (kinematics) · state assembly
  behavior/            behavioural features · crossing‑intention (heuristic + GRU)
  prediction/          multi‑modal trajectory prediction
  interaction/         conflict‑geometry metrics · interaction engine
  risk/                risk engine · near‑miss detection · explanation · counterfactuals
  ml/                  numpy NN (GRU+Adam) · datasets · baselines · training · evaluation · calibration
  graph/               event graph · city analyzer
  provenance/          content‑addressed model version · risk lineage records
  privacy/             in‑pipeline face + plate redaction
  security/            role‑based access control · append‑only audit log
  streaming/           camera‑fleet registry · bounded ingestion queue
  monitoring/          PSI/KS drift · calibration tracking
  pipeline.py          end‑to‑end orchestrator
  api/                 FastAPI dashboard + JSON API
frontend/index.html    self‑contained cinematic dashboard
scripts/               train_all · build_demo
tests/                 conflict geometry · homography · Kalman · GRU grad‑check · pipeline smoke
```

---

## Honesty

* Splits are **by scene**, so no pedestrian appears in both train and test.
* The risk score is **calibrated** (reliability diagram + ECE), not asserted.
* Counterfactuals are **model simulations** over recorded trajectories, not proof that any
  single intervention would have changed the outcome.
* Simulated results demonstrate the *method*; the same pipeline runs on PIE/JAAD via the
  adapter, where these numbers must be re‑earned on real footage.

MIT‑licensed.
