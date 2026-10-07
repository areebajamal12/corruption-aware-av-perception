import pytest

from av_perception.detection.evaluation import (
    average_precision,
    intersection_over_union,
    match_frame,
    summarize_metrics,
)
from av_perception.detection.types import BoundingBox, Detection, GroundTruth


def detection(class_name: str, confidence: float, box: BoundingBox) -> Detection:
    return Detection(class_name, confidence, box, class_name)


def truth(class_name: str, box: BoundingBox, token: str = "gt") -> GroundTruth:
    return GroundTruth(class_name, box, token, class_name, 4)


def test_iou_identical_boxes() -> None:
    box = BoundingBox(10, 10, 30, 30)
    assert intersection_over_union(box, box) == 1.0


def test_iou_partial_overlap() -> None:
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 15, 10)
    assert intersection_over_union(first, second) == pytest.approx(1 / 3)


def test_match_frame_is_class_aware_and_one_to_one() -> None:
    gt = [truth("vehicle", BoundingBox(0, 0, 10, 10))]
    predictions = [
        detection("vehicle", 0.9, BoundingBox(0, 0, 10, 10)),
        detection("vehicle", 0.8, BoundingBox(0, 0, 10, 10)),
        detection("pedestrian", 0.7, BoundingBox(0, 0, 10, 10)),
    ]
    matches = match_frame(predictions, gt, 0.5)
    assert [match.matched for match in matches] == [True, False, False]
    assert matches[0].ground_truth_index == 0


def test_average_precision_perfect_ranking() -> None:
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_average_precision_penalizes_early_false_positive() -> None:
    assert average_precision([0.9, 0.8], [False, True], 1) == pytest.approx(0.5)


def test_summary_counts_false_negatives() -> None:
    rows = [
        {"record_type": "detection", "class": "vehicle", "confidence": 0.9, "matched": True},
        {"record_type": "ground_truth", "class": "vehicle", "matched": True},
        {"record_type": "ground_truth", "class": "vehicle", "matched": False},
    ]
    result = summarize_metrics(rows, 0.5)
    assert result["per_class"]["vehicle"]["false_negatives"] == 1
    assert result["per_class"]["vehicle"]["recall"] == 0.5
