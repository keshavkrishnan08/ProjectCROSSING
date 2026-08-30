"""Role-based access control.

Four roles, each a superset of scopes. A request carries an API key (``X-API-Key``); the
key resolves to a :class:`Principal` whose role grants a set of scopes. Endpoints declare
the scope they need. An unknown or absent key resolves to the ``viewer`` role, which can see
only the de-identified operational view and never the raw feed, the lineage, or the audit
log.

The demo keys below are intentionally non-secret fixtures so the roles are exercisable; a
real deployment would resolve keys against an identity provider.
"""
from __future__ import annotations

from dataclasses import dataclass

# scope catalogue
SCOPES = {
    "feed:view": "See the de-identified operational view (risk, alerts, fleet).",
    "feed:raw": "See un-redacted imagery. Privileged.",
    "lineage:read": "Inspect the provenance chain behind a risk score.",
    "monitor:read": "See drift and calibration monitoring.",
    "audit:read": "Read the access audit log.",
    "feedback:write": "Submit human-in-the-loop corrections.",
}

# role -> granted scopes
ROLES = {
    "viewer":   {"feed:view"},
    "operator": {"feed:view", "monitor:read", "feedback:write"},
    "analyst":  {"feed:view", "monitor:read", "feedback:write", "lineage:read", "feed:raw"},
    "admin":    set(SCOPES),
}

# demo fixtures: key -> (subject, role)
_KEYS = {
    "crossing-operator-key": ("op@city.gov", "operator"),
    "crossing-analyst-key": ("analyst@city.gov", "analyst"),
    "crossing-admin-key": ("admin@city.gov", "admin"),
}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str

    @property
    def scopes(self) -> set:
        return ROLES.get(self.role, set())

    def can(self, scope: str) -> bool:
        return scope in self.scopes

    def as_dict(self) -> dict:
        return {"subject": self.subject, "role": self.role, "scopes": sorted(self.scopes)}


ANONYMOUS = Principal(subject="anonymous", role="viewer")


def resolve_principal(api_key: str | None) -> Principal:
    """Map an API key to a principal; unknown/empty keys fall back to the viewer role."""
    if api_key and api_key in _KEYS:
        subject, role = _KEYS[api_key]
        return Principal(subject=subject, role=role)
    return ANONYMOUS
