import numpy as np
import pytest

from av_perception.segmentation.evaluation import mask_iou, summarize_consistency


def test_identical_masks_have_perfect_iou() -> None:
    mask = np.array([[True, False], [True, True]])
    iou, intersection, union = mask_iou(mask, mask.copy())
    assert iou == 1.0
    assert intersection == 3
    assert union == 3


def test_mask_iou_partial_overlap() -> None:
    clean = np.array([[True, True], [False, False]])
    corrupted = np.array([[False, True], [True, False]])
    iou, intersection, union = mask_iou(clean, corrupted)
    assert iou == pytest.approx(1 / 3)
    assert intersection == 1
    assert union == 3


def test_two_empty_masks_are_consistent() -> None:
    empty = np.zeros((3, 3), dtype=bool)
    assert mask_iou(empty, empty)[0] == 1.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        mask_iou(np.zeros((2, 2)), np.zeros((3, 2)))


def test_consistency_summary() -> None:
    summary = summarize_consistency([0.2, 0.5, 0.8])
    assert summary["frames"] == 3
    assert summary["mean_iou"] == pytest.approx(0.5)
    assert summary["median_iou"] == pytest.approx(0.5)
    assert summary["p10_iou"] == pytest.approx(0.26)
