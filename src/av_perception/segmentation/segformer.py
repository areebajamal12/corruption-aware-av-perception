"""Cityscapes SegFormer PyTorch backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional

from av_perception.segmentation.base import ImageInput


class CityscapesSegFormer:
    """Extract the Cityscapes road class as a drivable-area proxy."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        device: str = "mps",
    ) -> None:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        self.model_name = model_name
        labels = {
            int(index): str(label).lower() for index, label in self.model.config.id2label.items()
        }
        try:
            self.road_class_id = next(index for index, label in labels.items() if label == "road")
        except StopIteration as error:
            raise ValueError(
                f"Model {model_name!r} has no 'road' semantic class: {labels}"
            ) from error

    @property
    def backend_name(self) -> str:
        return "segformer-b0-cityscapes-pytorch"

    @staticmethod
    def _load_rgb(image: ImageInput) -> tuple[Image.Image, tuple[int, int]]:
        if isinstance(image, Path):
            pil_image = Image.open(image).convert("RGB")
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        return pil_image, pil_image.size

    def segment_batch(self, images: Sequence[ImageInput]) -> list[np.ndarray]:
        if not images:
            return []
        loaded = [self._load_rgb(image) for image in images]
        pil_images = [item[0] for item in loaded]
        sizes = [item[1] for item in loaded]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            logits = self.model(pixel_values=pixel_values).logits
        masks: list[np.ndarray] = []
        for index, (width, height) in enumerate(sizes):
            resized = functional.interpolate(
                logits[index : index + 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            labels = resized.argmax(dim=1)[0].detach().cpu().numpy()
            masks.append(labels == self.road_class_id)
        for image in pil_images:
            image.close()
        return masks
