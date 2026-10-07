"""Common interface for replaceable detection backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from av_perception.detection.types import Detection

ImageInput = Path | np.ndarray


class ObjectDetector(Protocol):
    """Interface shared by PyTorch, ONNX, and future TensorRT backends."""

    @property
    def backend_name(self) -> str: ...

    def predict_batch(self, images: Sequence[ImageInput]) -> list[list[Detection]]:
        """Return one normalized prediction list for each input image."""
        ...
