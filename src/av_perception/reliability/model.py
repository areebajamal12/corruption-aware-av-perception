"""Calibrated lightweight runtime reliability estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def reliability_state(
    score: float, reliable_threshold: float = 0.8, degraded_threshold: float = 0.5
) -> str:
    if score >= reliable_threshold:
        return "reliable"
    if score >= degraded_threshold:
        return "degraded"
    return "unsafe"


@dataclass
class ReliabilityEstimator:
    """Logistic estimator with held-out isotonic probability calibration."""

    random_state: int = 20261006

    def __post_init__(self) -> None:
        self.classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        max_iter=2_000, class_weight="balanced", random_state=self.random_state
                    ),
                ),
            ]
        )
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        calibration_x: np.ndarray,
        calibration_y: np.ndarray,
    ) -> ReliabilityEstimator:
        self.feature_mean = np.mean(train_x, axis=0)
        self.feature_std = np.maximum(np.std(train_x, axis=0), 1e-6)
        train_distance = np.max(np.abs((train_x - self.feature_mean) / self.feature_std), axis=1)
        self.ood_threshold = float(np.quantile(train_distance, 0.995))
        self.classifier.fit(train_x, train_y)
        raw = self.classifier.predict_proba(calibration_x)[:, 1]
        self.calibrator.fit(raw, calibration_y)
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        raw = self.classifier.predict_proba(features)[:, 1]
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=float)
        distance = np.max(np.abs((features - self.feature_mean) / self.feature_std), axis=1)
        calibrated[distance > self.ood_threshold] = np.minimum(
            calibrated[distance > self.ood_threshold], 0.49
        )
        return calibrated
