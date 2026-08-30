"""Global configuration, physical constants, and coordinate conventions for CROSSING.

Coordinate systems
------------------
* **image space** — pixels, origin top-left, +x right, +y down. Raw detector output.
* **ground space** — metres on the road plane (a bird's-eye "digital twin"), origin at
  the scene's reference point, +x East, +y North. All physics (velocity, TTC, risk) is
  computed here after homography. Headings are radians, ``atan2(vy, vx)``, 0 = +x (East).

Everything downstream agrees on these two frames, so a real PIE clip and a simulated
scene are interchangeable once each provides an image->ground homography.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RECORDINGS_DIR = ARTIFACTS / "recordings"
MODELS_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"
FIGURES_DIR = ARTIFACTS / "figures"
DB_DIR = ARTIFACTS / "db"
DATA_DIR = ROOT / "data"

for _d in (RECORDINGS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, DB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Video / sampling
# --------------------------------------------------------------------------------------
FPS: float = 30.0
DT: float = 1.0 / FPS
IMAGE_W: int = 1920
IMAGE_H: int = 1080

# --------------------------------------------------------------------------------------
# Physical constants (SI units)
# --------------------------------------------------------------------------------------
G = 9.81
MPH_PER_MS = 2.2369362920544  # 1 m/s in mph

# Comfortable / emergency decelerations used by the counterfactual + IDM models.
COMFORT_DECEL = 2.5   # m/s^2
MAX_DECEL = 8.0       # m/s^2  (dry asphalt, panic braking)
REACTION_TIME = 0.9   # s      (human perception-reaction time)

# Nominal actor footprints (metres) for closest-point-of-approach geometry.
PED_RADIUS = 0.30
VEHICLE_LENGTH = 4.6
VEHICLE_WIDTH = 1.9
CYCLIST_LENGTH = 1.8
CYCLIST_WIDTH = 0.7


# --------------------------------------------------------------------------------------
# Prediction / interaction horizons
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Horizons:
    """Time horizons (seconds) used across prediction and risk."""

    observe: float = 1.5          # history length fed to learned models
    predict: tuple = (1.0, 2.0, 3.0, 5.0)
    predict_max: float = 5.0
    interaction_gate: float = 25.0  # only pair actors within this ground distance (m)


HORIZONS = Horizons()


# --------------------------------------------------------------------------------------
# Risk thresholds — deliberately conservative and *documented* rather than magic numbers.
# These define the operational near-miss label; the learned model is calibrated against it.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    # Time-to-collision bands (s)
    ttc_critical: float = 1.0
    ttc_high: float = 2.0
    ttc_warn: float = 4.0

    # Distance at closest point of approach (m) — how near the paths actually come.
    dcpa_critical: float = 0.8
    dcpa_high: float = 1.5
    dcpa_warn: float = 3.0

    # A near-miss requires a genuinely small predicted separation *and* real closing speed.
    nearmiss_dcpa: float = 1.2
    nearmiss_ttc: float = 2.0
    nearmiss_min_closing_speed: float = 2.0   # m/s

    # Evasive-signature thresholds (either party reacting hard is corroborating evidence).
    evasive_decel: float = 3.5                # m/s^2 vehicle braking
    evasive_ped_lat_acc: float = 1.5          # m/s^2 pedestrian lateral jerk-away

    # Risk score bands (0-100) for the UI.
    band_critical: float = 80.0
    band_high: float = 60.0
    band_medium: float = 35.0


RISK = RiskConfig()


# --------------------------------------------------------------------------------------
# Entity / class taxonomy
# --------------------------------------------------------------------------------------
PEDESTRIAN_CLASSES = {"person", "pedestrian"}
VEHICLE_CLASSES = {"car", "truck", "bus", "van"}
CYCLIST_CLASSES = {"bicycle", "motorcycle", "cyclist"}
INFRA_CLASSES = {"traffic_light", "crosswalk", "sign", "stop_sign"}

# COCO-ish detector label -> canonical CROSSING class.
CLASS_ALIASES = {
    "person": "person",
    "bicycle": "cyclist",
    "car": "car",
    "motorcycle": "cyclist",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
}


@dataclass
class Palette:
    """Palantir-adjacent operational palette shared by figures and the web UI."""

    bg: str = "#0a0e14"
    panel: str = "#0f1620"
    grid: str = "#1b2735"
    text: str = "#c9d6e2"
    person: str = "#43d1a0"
    vehicle: str = "#5aa2ff"
    cyclist: str = "#c792ea"
    infra: str = "#6b7d92"
    warn: str = "#ffb443"
    high: str = "#ff7a45"
    critical: str = "#ff3b6b"
    accent: str = "#38e0c8"


PALETTE = Palette()


@dataclass
class Config:
    """Top-level bundle passed through the pipeline; overridable per run."""

    fps: float = FPS
    seed: int = 7
    horizons: Horizons = field(default_factory=lambda: HORIZONS)
    risk: RiskConfig = field(default_factory=lambda: RISK)
    detector_noise_px: float = 4.0
    detector_dropout: float = 0.06         # per-frame miss probability
    detector_fp_rate: float = 0.01         # spurious detections / frame
    use_learned_intention: bool = True
    use_learned_trajectory: bool = True
    use_learned_risk: bool = True


DEFAULT_CONFIG = Config()
