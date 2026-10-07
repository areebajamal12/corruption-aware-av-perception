import numpy as np

from av_perception.demo import build_parser, runtime_feature_vector
from av_perception.detection.types import BoundingBox, Detection
from av_perception.reliability.features import FEATURE_NAMES
from av_perception.tracking.types import TrackedObject


def test_demo_defaults_to_noise_severity_four() -> None:
    args = build_parser().parse_args([])
    assert args.corruption == "noise"
    assert args.severity == 4
    assert args.output.name == "demo_noise_s4.gif"


def test_runtime_features_use_current_perception_outputs() -> None:
    image = np.full((10, 12, 3), 100, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=bool)
    mask[:, :6] = True
    detection = Detection("vehicle", 0.75, BoundingBox(1, 1, 8, 8), "car")
    track = TrackedObject(7, "vehicle", 0.75, detection.bbox, 0)

    features, ids = runtime_feature_vector(image, [detection], mask, [track], {7, 8})
    values = dict(zip(FEATURE_NAMES, features[0], strict=True))

    assert features.shape == (1, len(FEATURE_NAMES))
    assert values["detection_count"] == 1
    assert values["vehicle_count"] == 1
    assert values["predicted_road_fraction"] == 0.5
    assert values["track_continuity"] == 0.5
    assert ids == {7}
