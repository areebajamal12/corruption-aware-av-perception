"""Backend interface for drivable-area segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

ImageInput = Path | np.ndarray


class Segmenter(Protocol):
    """Interface implementable by PyTorch, ONNX, or TensorRT backends."""

    @property
    def backend_name(self) -> str: ...

    def segment_batch(self, images: Sequence[ImageInput]) -> list[np.ndarray]:
        """Return one boolean drivable-area mask at source resolution per image."""
        ...
