"""FastAPI backend for the CROSSING operational dashboard.

Serves the single-page dashboard and a small JSON API over the demo bundle and the event
database. The dashboard front-end (``frontend/index.html``) uses the same rendering code
as the self-contained Artifact, but fetches its data from ``/api/demo`` instead of having
it inlined.

Run:  ``crossing serve``  (or ``uvicorn crossing.api.server:app``)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import RECORDINGS_DIR, ROOT
from ..data.datamodel import EventDatabase
from ..security import AUDIT, SCOPES, resolve_principal

app = FastAPI(title="CROSSING", description="Pedestrian-vehicle safety intelligence")

_FRONTEND_DIR = ROOT / "frontend"
_DETECTION = _FRONTEND_DIR / "index.html"         # landing: real detection + predictive goal
_DASHBOARD = _FRONTEND_DIR / "dashboard.html"     # full cinematic prediction pipeline
_MEDIA = _FRONTEND_DIR / "media"
_DEMO = RECORDINGS_DIR / "demo_scene.json"

if _MEDIA.exists():
    app.mount("/media", StaticFiles(directory=str(_MEDIA)), name="media")


def _load_demo():
    if _DEMO.exists():
        return json.loads(_DEMO.read_text())
    return None


@app.get("/", response_class=HTMLResponse)
def index():
    """Landing page: real object detection on real video + the clear predictive goal."""
    if _DETECTION.exists():
        return _DETECTION.read_text()
    return "<h1>CROSSING</h1><p>Run <code>crossing demo</code> and <code>python -m scripts.detect_video</code>.</p>"


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """The full cinematic prediction pipeline over a processed scene."""
    if _DASHBOARD.exists():
        return _DASHBOARD.read_text()
    return "<h1>CROSSING</h1><p>Build the demo bundle first: <code>crossing demo</code></p>"


@app.get("/api/demo")
def demo():
    d = _load_demo()
    if d is None:
        raise HTTPException(404, "demo bundle not built — run `crossing demo`")
    return JSONResponse(d)


@app.get("/api/health")
def health():
    return {"status": "ok", "demo_available": _DEMO.exists()}


@app.get("/api/scenes")
def scenes():
    try:
        db = EventDatabase()
        return {"scenes": db.scenes(), "totals": db.totals()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/near_misses")
def near_misses(min_warning: float = 0.0):
    db = EventDatabase()
    return {"near_misses": db.near_misses(min_warning)}


@app.get("/api/hotspots")
def hotspots(top: int = 10):
    db = EventDatabase()
    return {"hotspots": db.dangerous_zones(top), "causes": db.cause_breakdown()}


class Feedback(BaseModel):
    scene_id: int
    event_id: int
    verdict: str            # 'confirmed' | 'false_positive' | 'missed'
    reason: str = ""
    reviewer: str = "analyst"


@app.post("/api/feedback")
def submit_feedback(fb: Feedback):
    """Human-in-the-loop: record an analyst's correction on an event (active learning)."""
    if fb.verdict not in ("confirmed", "false_positive", "missed"):
        raise HTTPException(400, "verdict must be confirmed | false_positive | missed")
    db = EventDatabase()
    fid = db.add_feedback(fb.scene_id, fb.event_id, fb.verdict, fb.reason, fb.reviewer)
    return {"id": fid, "summary": db.feedback_summary()}


@app.get("/api/feedback")
def list_feedback():
    db = EventDatabase()
    return {"summary": db.feedback_summary(), "labels": db.export_feedback()}


@app.get("/api/metrics")
def metrics():
    from ..config import METRICS_DIR
    out = {}
    for name in ("summary", "risk", "intention", "capabilities"):
        p = METRICS_DIR / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


# --------------------------------------------------------------------------------------
# Governance: role-based access control + audit over the sensitive operational surfaces.
# A request carries an ``X-API-Key`` header that resolves to a principal (role -> scopes).
# Every governed read is written to the audit log, whether it was allowed or denied.
# --------------------------------------------------------------------------------------
def _principal(x_api_key: str | None = Header(default=None)):
    return resolve_principal(x_api_key)


def require(scope: str, resource: str):
    def dep(request: Request, principal=Depends(_principal)):
        allowed = principal.can(scope)
        AUDIT.record(principal.subject, principal.role, "read", resource,
                     allowed=allowed, path=request.url.path, scope=scope)
        if not allowed:
            raise HTTPException(403, f"role '{principal.role}' lacks scope '{scope}'")
        return principal
    return dep


def _media_json(name: str):
    p = _MEDIA / name
    if not p.exists():
        raise HTTPException(404, f"{name} not built yet — run `python -m scripts.build_gallery`")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/whoami")
def whoami(x_api_key: str | None = Header(default=None)):
    """Resolve the caller's role and the scopes it grants (viewer if no/unknown key)."""
    return {"principal": resolve_principal(x_api_key).as_dict(), "scopes": SCOPES}


@app.get("/api/fleet")
def fleet(principal=Depends(require("feed:view", "fleet"))):
    """Fleet-wide camera status: scale, alerts, uptime, redaction policy."""
    return _media_json("fleet.json")


@app.get("/api/lineage")
def lineage(principal=Depends(require("lineage:read", "lineage"))):
    """Provenance chains behind flagged risk scores. Privileged (analyst+)."""
    return _media_json("lineage.json")


@app.get("/api/monitoring")
def monitoring(principal=Depends(require("monitor:read", "monitoring"))):
    """Live input-distribution drift + model calibration."""
    return _media_json("monitoring.json")


@app.get("/api/audit")
def audit(principal=Depends(require("audit:read", "audit"))):
    """The access audit log itself. Admin only."""
    return {"summary": AUDIT.summary(), "recent": AUDIT.tail(50)}
