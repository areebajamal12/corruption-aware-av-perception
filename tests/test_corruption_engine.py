import numpy as np
import pytest

from av_perception.corruption.engine import CORRUPTIONS, apply_corruption


@pytest.fixture
def image() -> np.ndarray:
    values = np.arange(48 * 64 * 3, dtype=np.uint32).reshape(48, 64, 3) % 256
    return values.astype(np.uint8)


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_corruptions_are_deterministic(image: np.ndarray, corruption: str) -> None:
    first = apply_corruption(image, corruption, severity=3, seed=1234)
    second = apply_corruption(image, corruption, severity=3, seed=1234)
    assert np.array_equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == np.uint8


@pytest.mark.parametrize("severity", [1, 2, 3, 4])
def test_every_severity_changes_image(image: np.ndarray, severity: int) -> None:
    for corruption in CORRUPTIONS:
        assert not np.array_equal(image, apply_corruption(image, corruption, severity, 99))


def test_invalid_severity_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Severity"):
        apply_corruption(image, "fog", severity=0, seed=1)
