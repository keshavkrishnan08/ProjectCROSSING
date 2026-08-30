"""Fleet registry and a bounded ingestion queue.

``FrameQueue`` is a fixed-capacity ring: when a consumer falls behind, the oldest frame is
dropped and a counter increments, so ingestion never blocks producers or grows without
bound. That dropped-frame count is itself a health signal surfaced per feed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field


class FrameQueue:
    """A bounded ring buffer expressing backpressure via frame drops."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._q: deque = deque()
        self.dropped = 0
        self.seen = 0

    def push(self, frame) -> bool:
        self.seen += 1
        if len(self._q) >= self.capacity:
            self._q.popleft()
            self.dropped += 1
            self._q.append(frame)
            return False
        self._q.append(frame)
        return True

    def pop(self):
        return self._q.popleft() if self._q else None

    @property
    def depth(self) -> int:
        return len(self._q)

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.seen if self.seen else 0.0


@dataclass
class CameraFeed:
    feed_id: str
    name: str
    lat: float
    lon: float
    fps: float = 30.0
    status: str = "online"          # online | degraded | offline
    peak_risk: int = 0              # 0..100
    at_risk: int = 0                # pedestrians currently flagged
    alerts_today: int = 0
    uptime: float = 0.999
    dropped_frames: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Fleet:
    def __init__(self, feeds: list[CameraFeed] | None = None):
        self.feeds: list[CameraFeed] = feeds or []

    def add(self, feed: CameraFeed):
        self.feeds.append(feed)

    def status(self) -> dict:
        online = sum(1 for f in self.feeds if f.status == "online")
        alerts = sum(f.alerts_today for f in self.feeds)
        at_risk = sum(f.at_risk for f in self.feeds)
        uptime = round(sum(f.uptime for f in self.feeds) / len(self.feeds), 4) if self.feeds else 0.0
        return {
            "cameras": len(self.feeds),
            "online": online,
            "degraded": sum(1 for f in self.feeds if f.status == "degraded"),
            "offline": sum(1 for f in self.feeds if f.status == "offline"),
            "alerts_today": alerts,
            "at_risk_now": at_risk,
            "fleet_uptime": uptime,
            "feeds": [f.as_dict() for f in self.feeds],
        }


# location fixtures (real-ish city coordinates) so the fleet map reads as a real deployment
_LOCATIONS = [
    (39.7439, -104.9899), (39.7392, -104.9903), (39.7508, -104.9964), (39.7286, -104.9897),
    (39.7621, -104.9812), (39.7340, -105.0012), (39.7555, -104.9700), (39.7190, -104.9942),
]


def build_fleet(scenes: list[dict]) -> Fleet:
    """Turn processed scene summaries into a fleet of camera feeds.

    Each ``scene`` provides ``name``/``scene``, ``peak_risk`` (0..100), ``at_risk`` and an
    optional ``alerts``. Degradation is inferred from a nonzero drop rate if supplied.
    """
    fleet = Fleet()
    for i, s in enumerate(scenes):
        lat, lon = _LOCATIONS[i % len(_LOCATIONS)]
        risk = int(s.get("peak_risk", s.get("risk", 0)))
        status = "online"
        if s.get("drop_rate", 0) > 0.15:
            status = "degraded"
        fleet.add(CameraFeed(
            feed_id=f"CAM-{i + 1:02d}",
            name=s.get("name", s.get("scene", f"Camera {i + 1}")),
            lat=lat, lon=lon,
            status=status,
            peak_risk=risk,
            at_risk=int(s.get("at_risk", 0)),
            alerts_today=int(s.get("alerts", 1 if risk >= 78 else 0)),
            uptime=float(s.get("uptime", 0.999)),
            dropped_frames=int(s.get("dropped_frames", 0)),
        ))
    return fleet
