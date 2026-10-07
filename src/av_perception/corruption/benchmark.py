"""Benchmark YOLO detection under deterministic camera corruptions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from av_perception.corruption.engine import CORRUPTIONS, apply_corruption
from av_perception.detection.base import ObjectDetector
from av_perception.detection.evaluation import match_frame, summarize_metrics
from av_perception.detection.ground_truth import project_ground_truth
from av_perception.detection.pipeline import (
    CSV_FIELDS,
    EvaluationConfig,
    FrameRecord,
    collect_frames,
    make_result_rows,
)
from av_perception.detection.yolo import UltralyticsYoloDetector


def derive_seed(base_seed: int, frame: FrameRecord, corruption: str, severity: int) -> int:
    """Derive a stable per-frame seed independent of Python hash randomization."""
    value = f"{base_seed}:{frame.sample_token}:{corruption}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def build_corrupted_batch(
    frames: list[FrameRecord],
    corruption: str,
    severity: int,
    base_seed: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate one in-memory BGR image per input frame."""
    images: list[np.ndarray] = []
    seeds: list[int] = []
    for frame in frames:
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise FileNotFoundError(frame.image_path)
        seed = derive_seed(base_seed, frame, corruption, severity)
        corrupted = apply_corruption(image, corruption, severity, seed)
        images.append(corrupted)
        seeds.append(seed)
    return images, seeds


def plot_ap_curves(metrics: dict[str, Any], destination: Path) -> None:
    """Plot clean and severity 1-4 AP for each corruption and class."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    clean = metrics["clean:0"]["per_class"]
    for axis, class_name in zip(axes, ("vehicle", "pedestrian"), strict=True):
        for corruption in CORRUPTIONS:
            values = [clean[class_name]["ap"]] + [
                metrics[f"{corruption}:{severity}"]["per_class"][class_name]["ap"]
                for severity in range(1, 5)
            ]
            axis.plot(range(5), values, marker="o", linewidth=2, label=corruption)
        axis.set_title(class_name.title())
        axis.set_xlabel("Severity (0 = clean)")
        axis.set_xticks(range(5))
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("AP@0.5")
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("YOLOv8s detection degradation under camera corruption")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_corruption_benchmark(
    nusc: Any,
    detector: ObjectDetector,
    output_dir: Path,
    config: EvaluationConfig,
    scene_limit: int | None,
    base_seed: int,
) -> dict[str, Any]:
    frames = collect_frames(nusc, scene_limit, config.channel)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    condition_metrics: dict[str, Any] = {}
    condition_timings: dict[str, Any] = {}
    conditions = [("clean", 0)] + [
        (corruption, severity) for corruption in CORRUPTIONS for severity in range(1, 5)
    ]

    for corruption, severity in tqdm(conditions, desc="Conditions"):
        condition_rows: list[dict[str, object]] = []
        started = time.perf_counter()
        for batch_start in range(0, len(frames), config.batch_size):
            batch = frames[batch_start : batch_start + config.batch_size]
            if corruption == "clean":
                image_paths = [frame.image_path for frame in batch]
                seeds = [0] * len(batch)
            else:
                image_paths, seeds = build_corrupted_batch(batch, corruption, severity, base_seed)
            predictions = detector.predict_batch(image_paths)
            for frame, detections, seed in zip(batch, predictions, seeds, strict=True):
                with Image.open(frame.image_path) as image:
                    width, height = image.size
                ground_truth = project_ground_truth(
                    nusc,
                    frame.sample_data_token,
                    width,
                    height,
                    config.min_visibility,
                    config.min_box_area,
                )
                matches = match_frame(detections, ground_truth, config.iou_threshold)
                condition_rows.extend(
                    make_result_rows(
                        frame,
                        detections,
                        ground_truth,
                        matches,
                        corruption=corruption,
                        severity=severity,
                        seed=seed,
                    )
                )
        elapsed = time.perf_counter() - started
        key = f"{corruption}:{severity}"
        condition_metrics[key] = summarize_metrics(condition_rows, config.iou_threshold)
        condition_timings[key] = {
            "elapsed_seconds": elapsed,
            "frames_per_second": len(frames) / elapsed if elapsed else 0.0,
            "milliseconds_per_frame": 1_000 * elapsed / len(frames) if frames else 0.0,
        }
        all_rows.extend(condition_rows)

    csv_path = output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "backend": detector.backend_name,
        "device": getattr(detector, "device", "unknown"),
        "scenes": len({frame.scene for frame in frames}),
        "frames_per_condition": len(frames),
        "conditions": len(conditions),
        "base_seed": base_seed,
        "config": asdict(config),
        "metrics": condition_metrics,
        "timings": condition_timings,
        "csv": str(csv_path.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_ap_curves(condition_metrics, output_dir / "ap_vs_severity.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8s under camera corruptions.")
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scenes", type=int, help="first N scenes; omit for all scenes")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-visibility", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--min-box-area", type=float, default=400.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261006)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot.resolve()), verbose=False)
    config = EvaluationConfig(
        min_visibility=args.min_visibility,
        min_box_area=args.min_box_area,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_visualizations=0,
    )
    detector = UltralyticsYoloDetector(
        weights=args.weights,
        device=args.device,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    result = run_corruption_benchmark(
        nusc, detector, args.output_dir, config, args.scenes, args.seed
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
