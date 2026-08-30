# CROSSING — design overview

**Observe → Understand → Predict → Act.** CROSSING turns street video into a
continuously-updated model of every person, vehicle, interaction, and emerging danger.

## Pipeline
```
video / sim  →  perception (detect+pose)  →  tracking (ByteTrack+Kalman, persistent IDs)
             →  world model (homography → ground metres; velocity/accel/heading/jerk)
             →  behavior (crossing intention, attention)  →  prediction (multi-modal trajectories)
             →  interaction engine (TTC / DCPA / TCPA / PET / path-conflict)
             →  risk engine (calibrated) + near-miss detection + explanation + counterfactuals
             →  event graph + city hotspots  →  operational UI
```
The object detector is just the sensor; the intelligence layer is the project.

## Ground truth
`crossing.data.simulator` generates signalised-intersection scenes with IDM vehicles,
gap-acceptance pedestrians, and seeded conflict scenarios (permissive-left, red-jaywalk,
occluded-emergence). Every scene is deterministic in its seed → leakage-free splits.
Real PIE/JAAD footage plugs in through `crossing.data.pie_adapter` (same schema).
