import pytest

from av_perception.detection.types import BoundingBox, Detection, GroundTruth
from av_perception.tracking.bytetrack import ByteTrackTracker
from av_perception.tracking.evaluation import identity_metrics, match_tracks
from av_perception.tracking.types import TrackedObject


def _gt(token: str, box: BoundingBox) -> GroundTruth:
    return GroundTruth("vehicle", box, f"ann-{token}", "vehicle.car", 4, token)


def _track(track_id: int, box: BoundingBox) -> TrackedObject:
    return TrackedObject(track_id, "vehicle", 0.9, box, 0)


def _row(frame: int, token: str, track_id: int | None) -> dict[str, object]:
    return {
        "record_type": "ground_truth",
        "scene": "scene-1",
        "frame": frame,
        "instance_token": token,
        "matched": track_id is not None,
        "matched_track_id": track_id if track_id is not None else "",
    }


def _pred(frame: int, track_id: int) -> dict[str, object]:
    return {"record_type": "track", "scene": "scene-1", "frame": frame, "track_id": track_id}


def test_match_tracks_is_class_aware_and_uses_iou() -> None:
    tracks = [_track(7, BoundingBox(0, 0, 10, 10))]
    truth = [_gt("instance-a", BoundingBox(0, 0, 10, 10))]
    assert match_tracks(tracks, truth, 0.5) == [(0, 0, pytest.approx(1.0))]


def test_identity_metrics_perfect_sequence() -> None:
    rows = [_row(0, "a", 1), _pred(0, 1), _row(1, "a", 1), _pred(1, 1)]
    result = identity_metrics(rows)
    assert result["idf1"] == 1.0
    assert result["id_switches"] == 0
    assert result["fragmentations"] == 0


def test_identity_metrics_switch_and_fragmentation() -> None:
    rows = [
        _row(0, "a", 1),
        _pred(0, 1),
        _row(1, "a", None),
        _row(2, "a", 2),
        _pred(2, 2),
    ]
    result = identity_metrics(rows)
    assert result["id_switches"] == 1
    assert result["fragmentations"] == 1
    assert result["observation_retention"] == pytest.approx(2 / 3)


def test_idf1_penalizes_identity_split() -> None:
    rows = [
        _row(0, "a", 1),
        _pred(0, 1),
        _row(1, "a", 2),
        _pred(1, 2),
    ]
    assert identity_metrics(rows)["idf1"] == pytest.approx(0.5)


def test_bytetrack_adapter_retains_id_for_stationary_detection() -> None:
    tracker = ByteTrackTracker()
    detection = Detection("vehicle", 0.9, BoundingBox(0, 0, 100, 100), "car")
    first = tracker.update([detection])
    second = tracker.update([detection])
    assert first[0].track_id == second[0].track_id
    assert second[0].class_name == "vehicle"


def test_bytetrack_adapter_handles_empty_frame() -> None:
    assert ByteTrackTracker().update([]) == []
