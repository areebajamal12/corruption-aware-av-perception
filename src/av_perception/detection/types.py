"""Backend-independent object detection types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned image-space box in xyxy pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            x1=min(max(self.x1, 0.0), float(width)),
            y1=min(max(self.y1, 0.0), float(height)),
            x2=min(max(self.x2, 0.0), float(width)),
            y2=min(max(self.y2, 0.0), float(height)),
        )


@dataclass(frozen=True)
class Detection:
    """A normalized detector prediction."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    source_class: str


@dataclass(frozen=True)
class GroundTruth:
    """A filtered nuScenes annotation projected into an image."""

    class_name: str
    bbox: BoundingBox
    annotation_token: str
    source_category: str
    visibility: int
    instance_token: str = ""


@dataclass(frozen=True)
class DetectionMatch:
    """A detection and its best one-to-one ground-truth match."""

    detection_index: int
    ground_truth_index: int | None
    iou: float
    matched: bool
