"""Identity matching and standard tracking metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from av_perception.detection.evaluation import match_frame
from av_perception.detection.types import Detection, GroundTruth
from av_perception.tracking.types import TrackedObject


def match_tracks(
    tracks: list[TrackedObject], ground_truth: list[GroundTruth], iou_threshold: float
) -> list[tuple[int, int, float]]:
    """Class-aware one-to-one IoU matching of tracks to ground truth."""
    detections = [
        Detection(track.class_name, track.confidence, track.bbox, "tracked") for track in tracks
    ]
    matches = match_frame(detections, ground_truth, iou_threshold)
    return [
        (match.detection_index, int(match.ground_truth_index), match.iou)
        for match in matches
        if match.matched and match.ground_truth_index is not None
    ]


def identity_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Compute IDF1, ID switches, fragmentations, and identity retention.

    IDF1 uses a global Hungarian assignment that maximizes matched observations between
    ground-truth instance tokens and predicted track IDs.
    """
    gt_rows = [row for row in rows if row["record_type"] == "ground_truth"]
    pred_rows = [row for row in rows if row["record_type"] == "track"]
    gt_total = len(gt_rows)
    pred_total = len(pred_rows)
    pairs = Counter(
        (str(row["instance_token"]), (str(row["scene"]), int(row["matched_track_id"])))
        for row in gt_rows
        if row["matched"]
    )
    gt_ids = sorted({pair[0] for pair in pairs})
    pred_ids = sorted({pair[1] for pair in pairs})
    idtp = 0
    if gt_ids and pred_ids:
        matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        gt_index = {value: index for index, value in enumerate(gt_ids)}
        pred_index = {value: index for index, value in enumerate(pred_ids)}
        for (gt_id, pred_id), count in pairs.items():
            matrix[gt_index[gt_id], pred_index[pred_id]] = count
        row_indices, column_indices = linear_sum_assignment(-matrix)
        idtp = int(matrix[row_indices, column_indices].sum())
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    denominator = 2 * idtp + idfp + idfn

    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        histories[str(row["instance_token"])].append(row)
    switches = 0
    fragmentations = 0
    retention: list[float] = []
    for observations in histories.values():
        observations.sort(key=lambda row: (str(row["scene"]), int(row["frame"])))
        matched_count = sum(bool(row["matched"]) for row in observations)
        retention.append(matched_count / len(observations))
        last_track: int | None = None
        seen_match = False
        gap_after_match = False
        for observation in observations:
            if observation["matched"]:
                track_id = int(observation["matched_track_id"])
                if last_track is not None and track_id != last_track:
                    switches += 1
                if seen_match and gap_after_match:
                    fragmentations += 1
                last_track = track_id
                seen_match = True
                gap_after_match = False
            elif seen_match:
                gap_after_match = True

    matched_observations = sum(bool(row["matched"]) for row in gt_rows)
    return {
        "ground_truth_observations": gt_total,
        "predicted_track_observations": pred_total,
        "matched_observations": matched_observations,
        "ground_truth_identities": len(histories),
        "predicted_identities": len({(row["scene"], row["track_id"]) for row in pred_rows}),
        "id_true_positives": idtp,
        "id_false_positives": idfp,
        "id_false_negatives": idfn,
        "idf1": (2 * idtp / denominator) if denominator else 0.0,
        "id_switches": switches,
        "fragmentations": fragmentations,
        "observation_retention": matched_observations / gt_total if gt_total else 0.0,
        "mean_identity_retention": float(np.mean(retention)) if retention else 0.0,
        "id_switches_per_100_gt": 100 * switches / gt_total if gt_total else 0.0,
        "fragmentations_per_100_gt": 100 * fragmentations / gt_total if gt_total else 0.0,
    }
