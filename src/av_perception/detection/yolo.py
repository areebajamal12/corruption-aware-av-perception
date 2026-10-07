"""Ultralytics YOLOv8 PyTorch backend."""

from __future__ import annotations

from collections.abc import Sequence

from av_perception.detection.base import ImageInput
from av_perception.detection.types import BoundingBox, Detection

COCO_CLASS_MAPPING = {
    0: "pedestrian",  # person
    2: "vehicle",  # car
    5: "vehicle",  # bus
    7: "vehicle",  # truck
}


class UltralyticsYoloDetector:
    """Adapt Ultralytics results into backend-independent detections."""

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        device: str = "mps",
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        batch_size: int = 8,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.batch_size = batch_size

    @property
    def backend_name(self) -> str:
        return "ultralytics-yolov8s-pytorch"

    def predict_batch(self, images: Sequence[ImageInput]) -> list[list[Detection]]:
        if not images:
            return []
        results = self._model.predict(
            source=[str(image) if not hasattr(image, "shape") else image for image in images],
            device=self.device,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            batch=self.batch_size,
            classes=sorted(COCO_CLASS_MAPPING),
            verbose=False,
        )
        normalized: list[list[Detection]] = []
        for result in results:
            frame: list[Detection] = []
            if result.boxes is not None:
                coordinates = result.boxes.xyxy.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                class_ids = result.boxes.cls.detach().cpu().to(int).tolist()
                for xyxy, confidence, class_id in zip(
                    coordinates, confidences, class_ids, strict=True
                ):
                    frame.append(
                        Detection(
                            class_name=COCO_CLASS_MAPPING[class_id],
                            confidence=float(confidence),
                            bbox=BoundingBox(*map(float, xyxy)),
                            source_class=str(result.names[class_id]),
                        )
                    )
            normalized.append(frame)
        return normalized
