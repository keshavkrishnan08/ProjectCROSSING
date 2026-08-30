"""Append-only audit log of sensitive-data access.

Every access to a governed resource is written as one JSON line: who (subject + role), what
(action + resource), the outcome (allowed / denied), and when. The log is append-only by
convention and cheap to tail for the operational view. Timestamps are UTC ISO-8601.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..config import DB_DIR


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else (DB_DIR / "audit.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, subject, role, action, resource, allowed=True, **extra):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "subject": subject, "role": role, "action": action,
            "resource": resource, "allowed": bool(allowed),
        }
        if extra:
            entry.update(extra)
        line = json.dumps(entry, separators=(",", ":"))
        with self._lock:
            with self.path.open("a") as fh:
                fh.write(line + "\n")
        return entry

    def entries(self):
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return out

    def tail(self, n=50):
        return self.entries()[-n:]

    def summary(self):
        rows = self.entries()
        by_action = Counter(r.get("action", "?") for r in rows)
        by_role = Counter(r.get("role", "?") for r in rows)
        denied = sum(1 for r in rows if not r.get("allowed", True))
        return {
            "total": len(rows),
            "denied": denied,
            "by_action": dict(by_action.most_common()),
            "by_role": dict(by_role.most_common()),
            "last_ts": rows[-1]["ts"] if rows else None,
        }


AUDIT = AuditLog()
