"""Backend-independent tracking types."""

from __future__ import annotations

from dataclasses import dataclass

from av_perception.detection.types import BoundingBox


@dataclass(frozen=True)
class TrackedObject:
    """A tracker output in image coordinates."""

    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    detection_index: int
