import numpy as np

from av_perception.reliability.features import image_quality_features
from av_perception.reliability.model import ReliabilityEstimator, reliability_state


def test_reliability_states() -> None:
    assert reliability_state(0.9) == "reliable"
    assert reliability_state(0.6) == "degraded"
    assert reliability_state(0.2) == "unsafe"


def test_image_features_are_finite() -> None:
    features = image_quality_features(np.full((20, 30, 3), 128, dtype=np.uint8))
    assert all(np.isfinite(value) for value in features.values())
    assert features["brightness_mean"] > 0


def test_estimator_returns_bounded_calibrated_scores() -> None:
    train_x = np.asarray([[0.0], [0.1], [0.9], [1.0]])
    train_y = np.asarray([0, 0, 1, 1])
    model = ReliabilityEstimator().fit(train_x, train_y, train_x, train_y)
    scores = model.predict_score(np.asarray([[0.05], [0.95]]))
    assert np.all((0 <= scores) & (scores <= 1))
    assert scores[0] < scores[1]


def test_estimator_caps_out_of_distribution_reliability() -> None:
    train_x = np.asarray([[0.0], [0.1], [0.9], [1.0]])
    train_y = np.asarray([0, 0, 1, 1])
    model = ReliabilityEstimator().fit(train_x, train_y, train_x, train_y)
    assert model.predict_score(np.asarray([[100.0]]))[0] <= 0.49
