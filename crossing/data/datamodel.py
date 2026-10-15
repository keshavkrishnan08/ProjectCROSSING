"""The operational event database, CROSSING's structured memory.

Everything the pipeline infers is persisted into a normalised SQLite schema so the platform
becomes an *event intelligence system* you can query, not a one-shot script:

    scenes → entities → entity_tracks
           → interactions (summarised) → risk_events → near_misses

with relationships mirrored from the schema:

    PERSON_17 --participates_in--> INTERACTION --involves--> VEHICLE_08
              --occurs_at--> zone --produces--> NEAR_MISS

This backs the API/analytics: "top dangerous zones", "all near-misses with >2 s warning",
"which vehicle manoeuvres dominate conflicts", etc., are plain SQL.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import DB_DIR
from ..schema import Recording


SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    scene_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT, fps REAL, n_frames INTEGER, duration REAL,
    n_entities INTEGER, n_interactions INTEGER, n_events INTEGER,
    n_near_misses INTEGER, mean_early_warning REAL, meta TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    scene_id INTEGER, entity_id INTEGER, kind TEXT, cls TEXT,
    first_frame INTEGER, last_frame INTEGER, n_frames INTEGER,
    max_risk REAL, n_interactions INTEGER, n_near_misses INTEGER,
    PRIMARY KEY (scene_id, entity_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_uid INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER, event_id INTEGER, a_id INTEGER, b_id INTEGER,
    band TEXT, is_near_miss INTEGER, peak_risk REAL, peak_t REAL,
    min_ttc REAL, min_dcpa REAL, max_closing REAL,
    early_warning REAL, zone TEXT, label TEXT,
    loc_x REAL, loc_y REAL, contributions TEXT, evasive TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_scene ON events(scene_id);
CREATE INDEX IF NOT EXISTS idx_events_zone  ON events(zone);
CREATE INDEX IF NOT EXISTS idx_events_nm    ON events(is_near_miss);
CREATE TABLE IF NOT EXISTS human_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER, event_id INTEGER,
    verdict TEXT,          -- 'confirmed' | 'false_positive' | 'missed'
    reason TEXT, reviewer TEXT, created_at TEXT
);
"""


class EventDatabase:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or (DB_DIR / "crossing.db"))
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def reset(self):
        self.conn.executescript(
            "DROP TABLE IF EXISTS scenes; DROP TABLE IF EXISTS entities; DROP TABLE IF EXISTS events;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- ingest -----------------------------------------------------------------------
    def ingest(self, rec: Recording) -> int:
        m = rec.meta
        cur = self.conn.execute(
            "INSERT INTO scenes(name,fps,n_frames,duration,n_entities,n_interactions,"
            "n_events,n_near_misses,mean_early_warning,meta) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (rec.scene.name, rec.fps, rec.n_frames, rec.duration,
             m.get("n_entities", 0), m.get("n_interactions", 0), m.get("n_events", 0),
             m.get("n_near_misses", 0), m.get("mean_early_warning_s", 0.0), json.dumps(m)))
        scene_id = cur.lastrowid

        # entity aggregates
        agg = {}
        for ws in rec.frames:
            for e in ws.entities:
                a = agg.setdefault(e.id, {"kind": e.kind.value, "cls": e.cls,
                                          "first": ws.frame, "last": ws.frame, "n": 0,
                                          "max_risk": 0.0, "inter": set()})
                a["last"] = ws.frame
                a["n"] += 1
                a["max_risk"] = max(a["max_risk"], e.risk_score)
                a["inter"].update(e.interaction_ids)
        nm_by_ent = {}
        for ev in rec.events:
            if ev.is_near_miss:
                nm_by_ent[ev.a_id] = nm_by_ent.get(ev.a_id, 0) + 1
                nm_by_ent[ev.b_id] = nm_by_ent.get(ev.b_id, 0) + 1
        for eid, a in agg.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO entities VALUES(?,?,?,?,?,?,?,?,?,?)",
                (scene_id, eid, a["kind"], a["cls"], a["first"], a["last"], a["n"],
                 round(a["max_risk"], 1), len(a["inter"]), nm_by_ent.get(eid, 0)))

        for ev in rec.events:
            loc = ev.location or [None, None]
            self.conn.execute(
                "INSERT INTO events(scene_id,event_id,a_id,b_id,band,is_near_miss,peak_risk,"
                "peak_t,min_ttc,min_dcpa,max_closing,early_warning,zone,label,loc_x,loc_y,"
                "contributions,evasive) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (scene_id, ev.id, ev.a_id, ev.b_id, ev.band.value, int(ev.is_near_miss),
                 ev.peak_risk, ev.peak_t,
                 None if ev.min_ttc == float("inf") else ev.min_ttc,
                 ev.min_dcpa, ev.max_closing_speed, ev.early_warning_s, ev.zone, ev.label,
                 loc[0], loc[1], json.dumps(ev.contributions), json.dumps(ev.evasive)))
        self.conn.commit()
        return scene_id

    # -- queries ----------------------------------------------------------------------
    def scenes(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM scenes ORDER BY scene_id")]

    def near_misses(self, min_warning: float = 0.0) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM events WHERE is_near_miss=1 AND early_warning>=? "
            "ORDER BY peak_risk DESC", (min_warning,))]

    def dangerous_zones(self, top: int = 10) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT zone, COUNT(*) AS events, "
            "SUM(is_near_miss) AS near_misses, ROUND(AVG(peak_risk),1) AS mean_risk, "
            "ROUND(AVG(early_warning),2) AS mean_warning "
            "FROM events WHERE zone IS NOT NULL GROUP BY zone "
            "ORDER BY near_misses DESC, mean_risk DESC LIMIT ?", (top,))]

    def cause_breakdown(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT label, COUNT(*) AS n, SUM(is_near_miss) AS near_misses "
            "FROM events GROUP BY label ORDER BY near_misses DESC")]

    # -- human-in-the-loop / active learning ------------------------------------------
    def add_feedback(self, scene_id: int, event_id: int, verdict: str,
                     reason: str = "", reviewer: str = "analyst",
                     created_at: str | None = None) -> int:
        """Record a human correction on an event (the seed of an active-learning loop).

        Confirmed / false-positive / missed labels accumulate here; ``export_feedback``
        turns them into supervised targets that a periodic retraining job folds back into
        the risk model, SYSTEM alert -> human verdict -> new label -> improved model.
        """
        cur = self.conn.execute(
            "INSERT INTO human_labels(scene_id,event_id,verdict,reason,reviewer,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (scene_id, event_id, verdict, reason, reviewer, created_at or ""))
        self.conn.commit()
        return cur.lastrowid

    def export_feedback(self) -> list[dict]:
        """Human labels joined to their event features, ready to retrain on."""
        rows = self.conn.execute(
            "SELECT h.verdict, h.reason, e.* FROM human_labels h "
            "JOIN events e ON e.scene_id=h.scene_id AND e.event_id=h.event_id").fetchall()
        return [dict(r) for r in rows]

    def feedback_summary(self) -> dict:
        rows = self.conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM human_labels GROUP BY verdict").fetchall()
        return {r["verdict"]: r["n"] for r in rows}

    def totals(self) -> dict:
        r = self.conn.execute(
            "SELECT COUNT(*) AS scenes, SUM(n_interactions) AS interactions, "
            "SUM(n_events) AS events, SUM(n_near_misses) AS near_misses, "
            "SUM(n_entities) AS entities FROM scenes").fetchone()
        return dict(r) if r else {}

    def close(self):
        self.conn.close()
