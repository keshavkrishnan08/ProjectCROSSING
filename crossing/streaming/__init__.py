"""Streaming ingestion across a fleet of camera feeds.

One clip is a demo; a deployment is dozens of cameras streaming at once. This package models
that: a :class:`Fleet` registry of :class:`CameraFeed` sources, each with a health status and
a rolling risk read, plus a bounded :class:`FrameQueue` that expresses backpressure so a slow
consumer drops frames instead of unbounded buffering. The fleet view is what an operations
centre watches.
"""
from .fleet import CameraFeed, Fleet, FrameQueue, build_fleet

__all__ = ["CameraFeed", "Fleet", "FrameQueue", "build_fleet"]
