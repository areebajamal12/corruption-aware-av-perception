"""Train and evaluate runtime perception reliability with scene-disjoint splits."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from av_perception.corruption.engine import CORRUPTIONS
from av_perception.reliability.features import FEATURE_NAMES, assemble_examples, feature_matrix
from av_perception.reliability.model import ReliabilityEstimator, reliability_state


@dataclass(frozen=True)
class ReliabilityConfig:
    quality_threshold: float = 0.6
    reliable_threshold: float = 0.8
    degraded_threshold: float = 0.5
    calibration_bins: int = 10
    random_state: int = 20261006


def scene_split(scenes: list[str]) -> dict[str, list[str]]:
    """Deterministic 60/10/30 scene split; no frame from a scene crosses splits."""
    ordered = sorted(scenes)
    if len(ordered) < 5:
        raise ValueError("At least five scenes are required for train/calibration/test splitting.")
    train_end = max(1, int(len(ordered) * 0.6))
    calibration_end = max(train_end + 1, int(len(ordered) * 0.7))
    return {
        "train": ordered[:train_end],
        "calibration": ordered[train_end:calibration_end],
        "test": ordered[calibration_end:],
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if index == bins - 1:
            mask |= scores == 1.0
        if np.any(mask):
            error += (
                np.sum(mask) / total * abs(float(np.mean(scores[mask]) - np.mean(labels[mask])))
            )
    return error


def evaluate_scores(
    labels: np.ndarray, scores: np.ndarray, config: ReliabilityConfig
) -> dict[str, Any]:
    states = [
        reliability_state(score, config.reliable_threshold, config.degraded_threshold)
        for score in scores
    ]
    unsafe = labels == 0
    false_safe = np.asarray(states) == "reliable"
    return {
        "examples": len(labels),
        "positive_rate": float(np.mean(labels)),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "expected_calibration_error": expected_calibration_error(
            labels, scores, config.calibration_bins
        ),
        "false_safe_rate": float(np.mean(false_safe[unsafe])) if np.any(unsafe) else 0.0,
        "state_counts": {
            state: states.count(state) for state in ("reliable", "degraded", "unsafe")
        },
    }


def _select(
    examples: list[dict[str, Any]],
    scenes: list[str],
    excluded_corruption: str | None = None,
    only_corruption: str | None = None,
) -> list[dict[str, Any]]:
    selected = [row for row in examples if row["scene"] in scenes]
    if excluded_corruption:
        selected = [row for row in selected if row["corruption"] != excluded_corruption]
    if only_corruption:
        selected = [row for row in selected if row["corruption"] == only_corruption]
    return selected


def _fit(
    train: list[dict[str, Any]], calibration: list[dict[str, Any]], config: ReliabilityConfig
) -> ReliabilityEstimator:
    return ReliabilityEstimator(config.random_state).fit(
        feature_matrix(train),
        np.asarray([row["acceptable"] for row in train]),
        feature_matrix(calibration),
        np.asarray([row["acceptable"] for row in calibration]),
    )


def plot_calibration(labels: np.ndarray, scores: np.ndarray, destination: Path) -> None:
    bins = np.linspace(0, 1, 11)
    predicted: list[float] = []
    observed: list[float] = []
    for index in range(10):
        mask = (scores >= bins[index]) & (scores < bins[index + 1])
        if index == 9:
            mask |= scores == 1.0
        if np.any(mask):
            predicted.append(float(np.mean(scores[mask])))
            observed.append(float(np.mean(labels[mask])))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    axis.plot(predicted, observed, marker="o", linewidth=2, label="estimator")
    axis.set(
        xlabel="Predicted reliability",
        ylabel="Observed acceptable rate",
        title="Held-out-scene reliability calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_heldout(metrics: dict[str, dict[str, Any]], destination: Path) -> None:
    names = list(metrics)
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title in zip(
        axes,
        ("auroc", "average_precision", "false_safe_rate"),
        ("AUROC", "Precision-recall AP", "False-safe rate"),
        strict=True,
    ):
        axis.bar(x, [metrics[name][metric] or 0.0 for name in names])
        axis.set_xticks(x, names, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Leave-one-corruption-out evaluation on held-out scenes")
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def run_experiment(
    detection_csv: Path,
    segmentation_csv: Path,
    tracking_csv: Path,
    output_dir: Path,
    config: ReliabilityConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = assemble_examples(
        detection_csv, segmentation_csv, tracking_csv, config.quality_threshold
    )
    splits = scene_split(sorted({row["scene"] for row in examples}))
    train = _select(examples, splits["train"])
    calibration = _select(examples, splits["calibration"])
    test = _select(examples, splits["test"])
    model = _fit(train, calibration, config)
    test_scores = model.predict_score(feature_matrix(test))
    test_labels = np.asarray([row["acceptable"] for row in test])
    test_metrics = evaluate_scores(test_labels, test_scores, config)

    rows: list[dict[str, Any]] = []
    for example, score in zip(test, test_scores, strict=True):
        rows.append(
            {
                "scene": example["scene"],
                "frame": example["frame"],
                "sample_token": example["sample_token"],
                "corruption": example["corruption"],
                "severity": example["severity"],
                "quality": example["quality"],
                "acceptable": example["acceptable"],
                "reliability_score": score,
                "state": reliability_state(
                    score, config.reliable_threshold, config.degraded_threshold
                ),
            }
        )
    with (output_dir / "reliability_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    heldout: dict[str, dict[str, Any]] = {}
    for corruption in CORRUPTIONS:
        heldout_train = _select(examples, splits["train"], excluded_corruption=corruption)
        heldout_calibration = _select(
            examples, splits["calibration"], excluded_corruption=corruption
        )
        heldout_test = _select(examples, splits["test"], only_corruption=corruption)
        heldout_model = _fit(heldout_train, heldout_calibration, config)
        scores = heldout_model.predict_score(feature_matrix(heldout_test))
        labels = np.asarray([row["acceptable"] for row in heldout_test])
        heldout[corruption] = evaluate_scores(labels, scores, config)

    joblib.dump(
        {"model": model, "features": FEATURE_NAMES, "config": asdict(config)},
        output_dir / "reliability_model.joblib",
    )
    plot_calibration(test_labels, test_scores, output_dir / "calibration.png")
    plot_heldout(heldout, output_dir / "heldout_corruptions.png")
    summary = {
        "config": asdict(config),
        "feature_names": FEATURE_NAMES,
        "scene_split": splits,
        "dataset_examples": len(examples),
        "train_examples": len(train),
        "calibration_examples": len(calibration),
        "test_examples": len(test),
        "test_metrics": test_metrics,
        "heldout_corruption_metrics": heldout,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate perception reliability.")
    parser.add_argument("--detection-csv", type=Path, required=True)
    parser.add_argument("--segmentation-csv", type=Path, required=True)
    parser.add_argument("--tracking-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ReliabilityConfig(quality_threshold=args.quality_threshold)
    print(
        json.dumps(
            run_experiment(
                args.detection_csv,
                args.segmentation_csv,
                args.tracking_csv,
                args.output_dir,
                config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
