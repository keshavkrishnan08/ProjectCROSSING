"""Tests for the governance layer: provenance, privacy, access control, fleet, monitoring."""
from __future__ import annotations

import numpy as np

from crossing.monitoring import (CalibrationTracker, ks_statistic,
                                  population_stability_index, drift_report)
from crossing.privacy import REDACTION_POLICY, redact_frame
from crossing.provenance import MODEL_VERSION, trace_risk
from crossing.security import AuditLog, resolve_principal
from crossing.streaming import FrameQueue, build_fleet


def test_model_version_is_content_addressed():
    assert MODEL_VERSION.startswith("histgb+depth@")
    assert len(MODEL_VERSION.split("@")[1]) == 10


def test_trace_risk_binds_score_to_inputs():
    hist = {
        7: {"cls": "person", "conf": [0.9, 0.95, 0.92], "first": 2, "last": 4},
        3: {"cls": "car", "conf": [0.8, 0.85], "first": 2, "last": 4},
    }
    rec = trace_risk(7, 3, 0.97, 4, {"d_real": 2.3, "align": 0.9, "prox": 0.8,
                                     "path": 1.0, "ttc_boost": 0.4}, hist).as_dict()
    assert rec["risk_pct"] == 97
    assert rec["pedestrian"]["track_id"] == 7
    assert rec["pedestrian"]["frames_seen"] == 3
    assert rec["driver"]["cls"] == "car"
    assert rec["components"]["d_real"] == 2.3
    assert rec["model_version"] == MODEL_VERSION


def test_redaction_alters_face_and_plate_regions_only():
    frame = np.full((200, 200, 3), 128, np.uint8)
    frame[:] = np.random.RandomState(0).randint(0, 255, frame.shape, np.uint8)
    before = frame.copy()
    peds = [{"box": [20, 20, 60, 120]}]      # head band = top 34%
    vehicles = [{"box": [100, 100, 180, 170]}]
    redact_frame(frame, peds, vehicles, REDACTION_POLICY)
    # a pixel in the pedestrian head band must have changed (mosaic applied)
    assert not np.array_equal(before[25:35, 30:50], frame[25:35, 30:50])
    # a pixel far from any box is untouched
    assert np.array_equal(before[0:10, 190:200], frame[0:10, 190:200])


def test_rbac_scopes_by_role():
    viewer = resolve_principal(None)
    analyst = resolve_principal("crossing-analyst-key")
    admin = resolve_principal("crossing-admin-key")
    assert viewer.role == "viewer" and not viewer.can("feed:raw")
    assert analyst.can("lineage:read") and analyst.can("feed:raw")
    assert not analyst.can("audit:read")
    assert admin.can("audit:read")


def test_audit_log_records_and_summarizes(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("a@x", "analyst", "read", "lineage", allowed=True)
    log.record("v@x", "viewer", "read", "lineage", allowed=False)
    s = log.summary()
    assert s["total"] == 2 and s["denied"] == 1
    assert s["by_action"]["read"] == 2


def test_frame_queue_backpressure_drops_oldest():
    q = FrameQueue(capacity=3)
    for i in range(10):
        q.push(i)
    assert q.depth == 3
    assert q.dropped == 7
    assert 0.6 < q.drop_rate < 0.8


def test_build_fleet_status():
    fleet = build_fleet([
        {"name": "A", "peak_risk": 97, "at_risk": 3},
        {"name": "B", "peak_risk": 10, "at_risk": 0},
    ])
    st = fleet.status()
    assert st["cameras"] == 2 and st["online"] == 2
    assert st["alerts_today"] == 1        # only the 97% scene raises an alert
    assert st["feeds"][0]["feed_id"] == "CAM-01"


def test_psi_and_ks_detect_shift():
    ref = np.concatenate([np.zeros(90), np.linspace(0, 1, 10)])
    same = ref.copy()
    shifted = np.concatenate([np.ones(90), np.linspace(0, 1, 10)])
    assert population_stability_index(ref, same) < 0.05
    assert population_stability_index(ref, shifted) > 0.25
    assert ks_statistic(ref, shifted) > ks_statistic(ref, same)


def test_drift_report_ranks_worst_feed():
    ref = list(np.zeros(80)) + list(np.linspace(0, 1, 20))
    rep = drift_report(ref, {"calm": ref, "busy": list(np.ones(50))})
    assert rep["feeds"][0]["feed"] == "busy"       # sorted worst-first
    assert rep["feeds"][0]["band"] == "drift"


def test_calibration_tracker_perfect_is_low_ece():
    tr = CalibrationTracker(bins=5)
    rng = np.random.RandomState(1)
    for _ in range(4000):
        p = rng.rand()
        tr.update(p, rng.rand() < p)     # outcomes drawn at the true probability
    assert tr.ece() < 0.05
    curve = tr.curve()
    assert len(curve) == 5
