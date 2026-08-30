"""Provenance and lineage: every derived value traces back to its inputs.

A risk score is never a bare number here. It carries the chain that produced it: which
detections, over which frames, at what confidence, tracked into which actor, interacting
with which vehicle, through which metric components, under which model version. This is the
Foundry-style guarantee that an operator can always answer "where did this 97 come from?".
"""
from .lineage import MODEL_VERSION, LineageRecord, TrackProvenance, model_version, trace_risk

__all__ = ["MODEL_VERSION", "LineageRecord", "TrackProvenance", "model_version", "trace_risk"]
