"""Explaining a risk score — *why* was this interaction dangerous?

A bare "risk = 91" is not actionable. CROSSING decomposes every score into the factors that
drove it (the :class:`PhysicsRisk` contributions), ranks them, renders them as shares, and
produces a one-line natural-language rationale. For the learned model, an optional
permutation attribution provides the same shape of explanation.
"""
from __future__ import annotations

import numpy as np

from .engine import PhysicsRisk, RISK_FEATURES


def decompose(contributions: dict) -> list[tuple[str, float, float]]:
    """Return [(factor, points, share), ...] sorted by |points| descending.

    ``points`` is the factor's contribution to the 0-100 score; ``share`` is its fraction
    of the total positive drive.
    """
    pos = {k: v for k, v in contributions.items() if v > 0}
    total = sum(pos.values()) or 1.0
    rows = [(k, v, v / total) for k, v in contributions.items()]
    return sorted(rows, key=lambda r: -abs(r[1]))


def rationale(contributions: dict, risk: float) -> str:
    rows = decompose(contributions)
    top = [r for r in rows if r[1] > 0][:3]
    if not top:
        return "Low risk: the actors are separating or well clear."
    parts = ", ".join(f"{name.lower()} ({pts:+.0f})" for name, pts, _ in top)
    mit = [r for r in rows if r[1] < 0]
    tail = f"; mitigated by {mit[0][0].lower()}" if mit else ""
    return f"Risk {risk:.0f}: driven by {parts}{tail}."


def permutation_attribution(learned, features: dict, n: int = 20, seed: int = 0) -> dict:
    """Model-agnostic attribution for the learned risk model.

    Perturbs each feature toward its neutral value and measures the drop in predicted
    probability — a fast, local Shapley-flavoured importance for the explanation panel.
    """
    rng = np.random.default_rng(seed)
    base = learned.proba(features)
    out = {}
    for k in RISK_FEATURES:
        f2 = dict(features)
        f2[k] = 0.0
        out[k] = round(100 * (base - learned.proba(f2)), 2)
    return out
