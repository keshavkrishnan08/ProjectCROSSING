"""Probability calibration and reliability diagnostics.

A safety score is only trustworthy if calibrated: among situations the model rates "80%",
about 80% should actually become near-misses. This module measures that (reliability curve
+ Expected Calibration Error) and provides an isotonic post-hoc calibrator used by the
learned risk model.
"""
from __future__ import annotations

import numpy as np


def reliability_curve(probs, labels, n_bins: int = 12):
    """Return per-bin (mean_predicted, mean_actual, count) for a reliability diagram."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if m.sum() == 0:
            out.append((float((lo + hi) / 2), None, 0))
        else:
            out.append((float(probs[m].mean()), float(labels[m].mean()), int(m.sum())))
    return out


def expected_calibration_error(probs, labels, n_bins: int = 12) -> float:
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    n = len(probs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for mean_p, mean_a, cnt in reliability_curve(probs, labels, n_bins):
        if cnt and mean_a is not None:
            ece += (cnt / n) * abs(mean_p - mean_a)
    return float(ece)


class IsotonicCalibrator:
    """Thin wrapper around sklearn's isotonic regression with safe extrapolation."""

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, probs, labels):
        self.ir.fit(np.asarray(probs, float), np.asarray(labels, float))
        self.fitted = True
        return self

    def predict(self, probs):
        p = np.asarray(probs, float)
        if not self.fitted:
            return p
        return self.ir.predict(p)
