"""Deterministic image corruptions with four documented severity levels."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

ImageArray = np.ndarray
Corruption = Callable[[ImageArray, int, np.random.Generator], ImageArray]


def _validate(image: ImageArray, severity: int) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 BGR image.")
    if severity not in {1, 2, 3, 4}:
        raise ValueError("Severity must be one of 1, 2, 3, or 4.")


def fog(image: ImageArray, severity: int, rng: np.random.Generator) -> ImageArray:
    """Blend spatially varying white haze into the image."""
    _validate(image, severity)
    alpha = (0.16, 0.30, 0.45, 0.60)[severity - 1]
    height, width = image.shape[:2]
    low_height, low_width = max(2, height // 32), max(2, width // 32)
    haze = rng.normal(0.0, 1.0, (low_height, low_width)).astype(np.float32)
    haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=max(low_width, low_height) / 25)
    haze -= haze.min()
    haze /= max(float(haze.max()), 1e-6)
    haze = cv2.resize(haze, (width, height), interpolation=cv2.INTER_CUBIC)
    local_alpha = np.clip(alpha * (0.8 + 0.4 * haze), 0.0, 0.85)[..., None]
    result = image.astype(np.float32) * (1.0 - local_alpha) + 255.0 * local_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def low_light(image: ImageArray, severity: int, rng: np.random.Generator) -> ImageArray:
    """Reduce exposure and add mild shot noise at stronger severities."""
    _validate(image, severity)
    del rng
    scale = (0.62, 0.44, 0.29, 0.17)[severity - 1]
    gamma = (1.15, 1.35, 1.60, 1.90)[severity - 1]
    normalized = image.astype(np.float32) / 255.0
    darkened = 255.0 * scale * np.power(normalized, gamma)
    return np.clip(darkened, 0, 255).astype(np.uint8)


def blur(image: ImageArray, severity: int, rng: np.random.Generator) -> ImageArray:
    """Apply increasingly strong Gaussian optical blur."""
    _validate(image, severity)
    del rng
    kernel = (5, 9, 15, 23)[severity - 1]
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0)


def noise(image: ImageArray, severity: int, rng: np.random.Generator) -> ImageArray:
    """Apply seeded additive Gaussian sensor noise."""
    _validate(image, severity)
    sigma = (8.0, 16.0, 28.0, 42.0)[severity - 1]
    perturbation = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + perturbation, 0, 255).astype(np.uint8)


def partial_occlusion(image: ImageArray, severity: int, rng: np.random.Generator) -> ImageArray:
    """Cover a seeded rectangular fraction of the camera image with near-black pixels."""
    _validate(image, severity)
    coverage = (0.08, 0.16, 0.28, 0.40)[severity - 1]
    height, width = image.shape[:2]
    aspect = float(rng.uniform(0.75, 1.35))
    occluder_width = min(width, max(1, int(np.sqrt(coverage * width * height * aspect))))
    occluder_height = min(height, max(1, int(coverage * width * height / occluder_width)))
    x1 = int(rng.integers(0, max(1, width - occluder_width + 1)))
    y1 = int(rng.integers(0, max(1, height - occluder_height + 1)))
    output = image.copy()
    output[y1 : y1 + occluder_height, x1 : x1 + occluder_width] = 8
    return output


CORRUPTIONS: dict[str, Corruption] = {
    "fog": fog,
    "low_light": low_light,
    "blur": blur,
    "noise": noise,
    "partial_occlusion": partial_occlusion,
}


def apply_corruption(image: ImageArray, corruption: str, severity: int, seed: int) -> ImageArray:
    """Apply a named corruption reproducibly from an explicit seed."""
    try:
        implementation = CORRUPTIONS[corruption]
    except KeyError as error:
        raise ValueError(
            f"Unknown corruption {corruption!r}; choose from {sorted(CORRUPTIONS)}"
        ) from error
    return implementation(image, severity, np.random.default_rng(seed))
