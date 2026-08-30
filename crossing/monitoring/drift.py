"""Distribution drift and calibration tracking.

``population_stability_index`` and ``ks_statistic`` compare a reference distribution (what the
model was trained/tested on) against what a live feed is actually producing. PSI has the usual
operational bands: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant drift.
"""
from __future__ import annotations

import numpy as np


def _hist(x, edges):
    c, _ = np.histogram(np.asarray(x, float), bins=edges)
    p = c.astype(float)
    p = p / p.sum() if p.sum() else p
    return np.clip(p, 1e-6, None)


def population_stability_index(reference, current, bins=10, lo=0.0, hi=1.0):
    """PSI between a reference and current sample over a fixed support [lo, hi]."""
    edges = np.linspace(lo, hi, bins + 1)
    r, c = _hist(reference, edges), _hist(current, edges)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_band(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "drift"


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), dependency-free."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    grid.sort()
    ca = np.searchsorted(a, grid, side="right") / max(len(a), 1)
    cb = np.searchsorted(b, grid, side="right") / max(len(b), 1)
    return float(np.max(np.abs(ca - cb))) if len(grid) else 0.0


def drift_report(reference, feeds: dict, bins=10):
    """PSI/KS of each live feed's score distribution against the reference distribution.

    ``feeds`` maps a feed label to a list of live scores (0..1). Returns per-feed drift with a
    band, plus reference/live histograms for a distribution-shift chart.
    """
    edges = np.linspace(0, 1, bins + 1)
    ref_hist = _hist(reference, edges).tolist()
    rows = []
    for label, scores in feeds.items():
        if not scores:
            continue
        psi = population_stability_index(reference, scores, bins=bins)
        rows.append({
            "feed": label,
            "psi": round(psi, 3),
            "ks": round(ks_statistic(reference, scores), 3),
            "band": psi_band(psi),
            "n": len(scores),
            "hist": _hist(scores, edges).tolist(),
        })
    rows.sort(key=lambda r: -r["psi"])
    return {
        "bins": bins,
        "reference_hist": ref_hist,
        "feeds": rows,
        "worst_band": rows[0]["band"] if rows else "stable",
    }


def temporal_drift_report(streams: dict, bins=10, min_len=8):
    """Per-feed drift over time: compare the first half of each feed's score stream to its
    second half. This flags a feed whose risk distribution is *shifting* while it runs, which
    is the real drift question, rather than merely flagging calm feeds as different from busy
    ones. A feed that stays consistent (calm or busy) reads as stable.
    """
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for label, stream in streams.items():
        n = len(stream)
        if n < min_len:
            continue
        h = n // 2
        ref, cur = stream[:h], stream[h:]
        psi = population_stability_index(ref, cur, bins=bins)
        rows.append({
            "feed": label, "psi": round(psi, 3), "ks": round(ks_statistic(ref, cur), 3),
            "band": psi_band(psi), "n": n, "hist": _hist(cur, edges).tolist(),
        })
    rows.sort(key=lambda r: -r["psi"])
    return {
        "bins": bins, "feeds": rows, "window": "first half vs second half of each feed",
        "worst_band": rows[0]["band"] if rows else "stable",
    }


class CalibrationTracker:
    """Bin predicted probabilities against realized outcomes to watch calibration.

    Feed it (predicted_prob, outcome_bool) pairs; ``curve()`` returns per-bin
    (mean_predicted, observed_rate, count) and ``ece()`` the expected calibration error.
    """

    def __init__(self, bins=10):
        self.bins = bins
        self.pred = []
        self.obs = []

    def update(self, p, y):
        self.pred.append(float(p))
        self.obs.append(1.0 if y else 0.0)

    def extend(self, pairs):
        for p, y in pairs:
            self.update(p, y)
        return self

    def curve(self):
        p = np.asarray(self.pred, float)
        y = np.asarray(self.obs, float)
        edges = np.linspace(0, 1, self.bins + 1)
        out = []
        for i in range(self.bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < self.bins - 1 else p <= edges[i + 1])
            if m.sum() == 0:
                out.append([round((edges[i] + edges[i + 1]) / 2, 3), None, 0])
            else:
                out.append([round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())])
        return out

    def ece(self):
        p = np.asarray(self.pred, float)
        y = np.asarray(self.obs, float)
        if len(p) == 0:
            return 0.0
        edges = np.linspace(0, 1, self.bins + 1)
        e = 0.0
        for i in range(self.bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < self.bins - 1 else p <= edges[i + 1])
            if m.sum():
                e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
        return float(e)
