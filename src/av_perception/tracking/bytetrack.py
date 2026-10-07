"""Adapter for the established Ultralytics ByteTrack implementation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from av_perception.detection.types import BoundingBox, Detection
from av_perception.tracking.types import TrackedObject

CLASS_TO_ID = {"pedestrian": 0, "vehicle": 1}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}


class ByteTrackTracker:
    """Run Ultralytics ByteTrack over normalized detector outputs."""

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 6,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        self._args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self.reset()

    @property
    def backend_name(self) -> str:
        return "ultralytics-bytetrack"

    def reset(self) -> None:
        self._tracker = BYTETracker(self._args)

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        values = np.asarray(
            [
                [
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                    detection.confidence,
                    CLASS_TO_ID[detection.class_name],
                ]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        boxes = Boxes(values, orig_shape=(1, 1))
        output = self._tracker.update(boxes)
        return [
            TrackedObject(
                track_id=int(row[4]),
                class_name=ID_TO_CLASS[int(row[6])],
                confidence=float(row[5]),
                bbox=BoundingBox(*map(float, row[:4])),
                detection_index=int(row[7]),
            )
            for row in output
        ]
