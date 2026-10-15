"""CROSSING, a real-time pedestrian-vehicle safety intelligence platform.

    Observe -> Understand -> Predict -> Act

Turn ordinary street video into a continuously updated model of every person, vehicle,
interaction, risk event, and emerging danger in an environment. The object detector is
just the sensor; the intelligence layer is the project.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import config, schema  # noqa: F401

__all__ = ["config", "schema", "__version__"]
