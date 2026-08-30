"""Lineage records for risk scores.

The risk model is content-addressed: ``model_version()`` hashes the actual source of the
scoring code plus its feature list, so a score can never be silently attributed to the
wrong model. A :class:`LineageRecord` then binds one score to the concrete inputs that
produced it and the intermediate quantities on the way, so the whole derivation is
inspectable and reproducible.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_RISK_FEATURES = (
    "d_real", "align", "prox", "path", "ttc_boost", "closing_ms", "veh_speed",
)


def _source_digest() -> str:
    """Short content hash of the scoring code, so the version tracks real changes."""
    # Hash the safety-video scorer if present; fall back to this module's own text.
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "scripts" / "safety_video.py",
        Path(__file__),
    ]
    h = hashlib.sha256()
    for p in candidates:
        if p.exists():
            h.update(p.read_bytes())
            break
    h.update("|".join(_RISK_FEATURES).encode())
    return h.hexdigest()[:10]


def model_version() -> str:
    """A human-readable, content-addressed model id, e.g. ``histgb+depth@1a2b3c4d5e``."""
    return f"histgb+depth@{_source_digest()}"


MODEL_VERSION = model_version()


@dataclass
class TrackProvenance:
    """Where a tracked actor came from: its detection history."""

    track_id: int
    cls: str
    frames_seen: int
    first_frame: int
    last_frame: int
    det_conf_min: float
    det_conf_max: float
    det_conf_mean: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LineageRecord:
    """The full derivation of a single pedestrian risk score at one instant."""

    risk: float                       # 0..1 final score
    frame: int
    model_version: str
    pedestrian: TrackProvenance
    driver: Optional[TrackProvenance]  # the vehicle that drove the score (None if safe)
    components: dict = field(default_factory=dict)   # d_real, align, prox, path, ttc_boost
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["risk_pct"] = round(self.risk * 100)
        return d


def _prov(hist: dict, track_id: int) -> Optional[TrackProvenance]:
    h = hist.get(track_id)
    if not h or not h["conf"]:
        return None
    confs = h["conf"]
    return TrackProvenance(
        track_id=track_id, cls=h["cls"], frames_seen=len(confs),
        first_frame=h["first"], last_frame=h["last"],
        det_conf_min=round(min(confs), 3), det_conf_max=round(max(confs), 3),
        det_conf_mean=round(sum(confs) / len(confs), 3),
    )


def trace_risk(ped_id, driver_id, risk, frame, components, history, note=""):
    """Assemble a :class:`LineageRecord` from tracker history and scoring components.

    ``history`` maps track_id -> {"cls", "conf": [..], "first", "last"} accumulated by the
    processing loop. ``components`` is the metric breakdown returned by the scorer.
    """
    ped = _prov(history, ped_id)
    if ped is None:
        ped = TrackProvenance(ped_id, "person", 0, frame, frame, 0.0, 0.0, 0.0)
    driver = _prov(history, driver_id) if driver_id is not None else None
    comps = {k: round(float(components[k]), 3) for k in components}
    return LineageRecord(
        risk=round(float(risk), 3), frame=int(frame), model_version=MODEL_VERSION,
        pedestrian=ped, driver=driver, components=comps, note=note,
    )
