"""Access control and audit.

Two ideas Palantir treats as table stakes: nobody sees more than their role allows, and
every read of sensitive data is recorded. :mod:`rbac` defines the roles and the scopes each
grants; :mod:`audit` appends an immutable-by-convention log of who accessed what, when.
"""
from .rbac import ROLES, SCOPES, Principal, resolve_principal
from .audit import AuditLog, AUDIT

__all__ = ["ROLES", "SCOPES", "Principal", "resolve_principal", "AuditLog", "AUDIT"]
