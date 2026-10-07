"""Replaceable multi-object tracker interface."""

from __future__ import annotations

from typing import Protocol

from av_perception.detection.types import Detection
from av_perception.tracking.types import TrackedObject


class MultiObjectTracker(Protocol):
    @property
    def backend_name(self) -> str: ...

    def reset(self) -> None: ...

    def update(self, detections: list[Detection]) -> list[TrackedObject]: ...
