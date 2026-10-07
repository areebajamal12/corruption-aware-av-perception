"""Evaluate drivable-area mask robustness under deterministic corruptions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from av_perception.corruption.benchmark import build_corrupted_batch
from av_perception.corruption.engine import CORRUPTIONS
from av_perception.detection.pipeline import FrameRecord, collect_frames
from av_perception.segmentation.base import Segmenter
from av_perception.segmentation.evaluation import mask_iou, summarize_consistency
from av_perception.segmentation.segformer import CityscapesSegFormer

CSV_FIELDS = [
    "scene",
    "frame",
    "sample_token",
    "image_path",
    "corruption",
    "severity",
    "seed",
    "mask_iou",
    "intersection_pixels",
    "union_pixels",
    "clean_road_pixels",
    "corrupted_road_pixels",
]


@dataclass(frozen=True)
class SegmentationConfig:
    channel: str = "CAM_FRONT"
    batch_size: int = 4
    base_seed: int = 20261006
    max_visualizations: int = 8


def save_mask_overlay(frame: FrameRecord, mask: np.ndarray, destination: Path) -> None:
    """Save a qualitative clean-mask overlay without implying ground-truth accuracy."""
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise FileNotFoundError(frame.image_path)
    overlay = image.copy()
    overlay[mask] = (40, 210, 40)
    blended = cv2.addWeighted(image, 0.62, overlay, 0.38, 0)
    cv2.putText(
        blended,
        "Predicted road mask (qualitative only; no camera pixel GT)",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), blended):
        raise RuntimeError(f"Could not write {destination}")


def plot_consistency(metrics: dict[str, dict[str, float | int]], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for corruption in CORRUPTIONS:
        values = [1.0] + [
            float(metrics[f"{corruption}:{severity}"]["mean_iou"]) for severity in range(1, 5)
        ]
        axis.plot(range(5), values, marker="o", linewidth=2, label=corruption)
    axis.set_title("Drivable-area segmentation robustness under camera corruption")
    axis.set_xlabel("Severity (0 = clean reference)")
    axis.set_ylabel("Mask IoU vs clean prediction")
    axis.set_xticks(range(5))
    axis.set_ylim(0.0, 1.03)
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _row(
    frame: FrameRecord,
    corruption: str,
    severity: int,
    seed: int,
    clean_mask: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    iou, intersection, union = mask_iou(clean_mask, mask)
    return {
        "scene": frame.scene,
        "frame": frame.frame,
        "sample_token": frame.sample_token,
        "image_path": str(frame.image_path),
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "mask_iou": iou,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "clean_road_pixels": int(clean_mask.sum()),
        "corrupted_road_pixels": int(mask.sum()),
    }


def run_segmentation_benchmark(
    nusc: Any,
    segmenter: Segmenter,
    output_dir: Path,
    config: SegmentationConfig,
    scene_limit: int | None,
) -> dict[str, Any]:
    frames = collect_frames(nusc, scene_limit, config.channel)
    conditions = [(corruption, severity) for corruption in CORRUPTIONS for severity in range(1, 5)]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    values: defaultdict[str, list[float]] = defaultdict(list)
    visualizations = 0
    visualization_stride = max(1, len(frames) // max(config.max_visualizations, 1))
    started = time.perf_counter()

    for batch_start in tqdm(range(0, len(frames), config.batch_size), desc="Segmenting"):
        batch = frames[batch_start : batch_start + config.batch_size]
        clean_masks = segmenter.segment_batch([frame.image_path for frame in batch])
        for offset, (frame, clean_mask) in enumerate(zip(batch, clean_masks, strict=True)):
            rows.append(_row(frame, "clean", 0, 0, clean_mask, clean_mask))
            values["clean:0"].append(1.0)
            index = batch_start + offset
            if visualizations < config.max_visualizations and index % visualization_stride == 0:
                save_mask_overlay(
                    frame,
                    clean_mask,
                    output_dir / "visualizations" / f"{frame.scene}_{frame.frame:03d}.jpg",
                )
                visualizations += 1

        for corruption, severity in conditions:
            corrupted_images, seeds = build_corrupted_batch(
                batch, corruption, severity, config.base_seed
            )
            masks = segmenter.segment_batch(corrupted_images)
            for frame, clean_mask, mask, seed in zip(batch, clean_masks, masks, seeds, strict=True):
                row = _row(frame, corruption, severity, seed, clean_mask, mask)
                rows.append(row)
                values[f"{corruption}:{severity}"].append(float(row["mask_iou"]))

    elapsed = time.perf_counter() - started
    csv_path = output_dir / "segmentation_consistency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        key: summarize_consistency(condition_values) for key, condition_values in values.items()
    }
    summary = {
        "backend": segmenter.backend_name,
        "device": getattr(segmenter, "device", "unknown"),
        "metric_scope": "clean-to-corrupted mask consistency; robustness, not accuracy",
        "scenes": len({frame.scene for frame in frames}),
        "frames": len(frames),
        "conditions": 21,
        "elapsed_seconds": elapsed,
        "frames_evaluated": len(rows),
        "config": asdict(config),
        "metrics": metrics,
        "csv": str(csv_path.resolve()),
        "visualizations": visualizations,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_consistency(metrics, output_dir / "mask_iou_vs_severity.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark drivable-mask robustness on nuScenes.")
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scenes", type=int, help="first N scenes; omit for all scenes")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--max-visualizations", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot.resolve()), verbose=False)
    segmenter = CityscapesSegFormer(model_name=args.model, device=args.device)
    config = SegmentationConfig(
        batch_size=args.batch_size,
        base_seed=args.seed,
        max_visualizations=args.max_visualizations,
    )
    summary = run_segmentation_benchmark(nusc, segmenter, args.output_dir, config, args.scenes)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
