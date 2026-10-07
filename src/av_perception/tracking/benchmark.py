"""Benchmark ByteTrack identity stability on clean and corrupted nuScenes keyframes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from av_perception.corruption.benchmark import build_corrupted_batch
from av_perception.corruption.engine import CORRUPTIONS
from av_perception.detection.base import ObjectDetector
from av_perception.detection.ground_truth import project_ground_truth
from av_perception.detection.pipeline import FrameRecord, collect_frames
from av_perception.detection.types import GroundTruth
from av_perception.detection.yolo import UltralyticsYoloDetector
from av_perception.tracking.base import MultiObjectTracker
from av_perception.tracking.bytetrack import ByteTrackTracker
from av_perception.tracking.evaluation import identity_metrics, match_tracks
from av_perception.tracking.types import TrackedObject

CSV_FIELDS = [
    "scene",
    "frame",
    "corruption",
    "severity",
    "seed",
    "sample_token",
    "record_type",
    "class",
    "track_id",
    "instance_token",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "iou",
    "matched",
    "matched_track_id",
    "matched_instance_token",
    "visibility",
]


@dataclass(frozen=True)
class TrackingConfig:
    channel: str = "CAM_FRONT"
    min_visibility: int = 2
    min_box_area: float = 400.0
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.25
    image_size: int = 640
    batch_size: int = 8
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 6
    match_thresh: float = 0.8
    max_visualizations: int = 12


def _rows_for_frame(
    frame: FrameRecord,
    tracks: list[TrackedObject],
    ground_truth: list[GroundTruth],
    matches: list[tuple[int, int, float]],
    corruption: str,
    severity: int,
    seed: int,
) -> list[dict[str, object]]:
    track_matches = {track_index: (gt_index, iou) for track_index, gt_index, iou in matches}
    gt_matches = {gt_index: (track_index, iou) for track_index, gt_index, iou in matches}
    common = {
        "scene": frame.scene,
        "frame": frame.frame,
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "sample_token": frame.sample_token,
    }
    rows: list[dict[str, object]] = []
    for index, track in enumerate(tracks):
        match = track_matches.get(index)
        truth = ground_truth[match[0]] if match else None
        rows.append(
            {
                **common,
                "record_type": "track",
                "class": track.class_name,
                "track_id": track.track_id,
                "instance_token": "",
                "confidence": track.confidence,
                "x1": track.bbox.x1,
                "y1": track.bbox.y1,
                "x2": track.bbox.x2,
                "y2": track.bbox.y2,
                "iou": match[1] if match else 0.0,
                "matched": bool(match),
                "matched_track_id": "",
                "matched_instance_token": truth.instance_token if truth else "",
                "visibility": "",
            }
        )
    for index, truth in enumerate(ground_truth):
        match = gt_matches.get(index)
        track = tracks[match[0]] if match else None
        rows.append(
            {
                **common,
                "record_type": "ground_truth",
                "class": truth.class_name,
                "track_id": "",
                "instance_token": truth.instance_token,
                "confidence": "",
                "x1": truth.bbox.x1,
                "y1": truth.bbox.y1,
                "x2": truth.bbox.x2,
                "y2": truth.bbox.y2,
                "iou": match[1] if match else 0.0,
                "matched": bool(match),
                "matched_track_id": track.track_id if track else "",
                "matched_instance_token": "",
                "visibility": truth.visibility,
            }
        )
    return rows


def draw_tracking(
    image: Any,
    tracks: list[TrackedObject],
    ground_truth: list[GroundTruth],
    matches: list[tuple[int, int, float]],
    destination: Path,
    label: str,
) -> None:
    canvas = cv2.imread(str(image)) if isinstance(image, Path) else image.copy()
    if canvas is None:
        raise FileNotFoundError(image)
    matched_tracks = {item[0] for item in matches}
    for index, track in enumerate(tracks):
        box = track.bbox
        color = (255, 160, 0) if index in matched_tracks else (0, 0, 255)
        cv2.rectangle(canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        cv2.putText(
            canvas,
            f"{track.class_name} T{track.track_id}",
            (int(box.x1), max(16, int(box.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    for truth in ground_truth:
        box = truth.bbox
        cv2.rectangle(
            canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 220, 0), 1
        )
    cv2.putText(
        canvas,
        f"{label} | tracks: blue/red | GT: green",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"Failed to write {destination}")


def plot_metrics(metrics: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    clean = metrics["clean:0"]
    for corruption in CORRUPTIONS:
        keys = [f"{corruption}:{severity}" for severity in range(1, 5)]
        axes[0].plot(
            range(5),
            [clean["idf1"]] + [metrics[key]["idf1"] for key in keys],
            marker="o",
            label=corruption,
        )
        axes[1].plot(
            range(5),
            [clean["id_switches_per_100_gt"]]
            + [metrics[key]["id_switches_per_100_gt"] for key in keys],
            marker="o",
            label=corruption,
        )
    axes[0].set_ylabel("IDF1")
    axes[1].set_ylabel("ID switches per 100 GT observations")
    for axis in axes:
        axis.set_xlabel("Severity (0 = clean)")
        axis.set_xticks(range(5))
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.suptitle("ByteTrack identity stability under camera corruption (2 Hz keyframes)")
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run_tracking_benchmark(
    nusc: Any,
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    output_dir: Path,
    config: TrackingConfig,
    scene_limit: int | None,
    base_seed: int,
) -> dict[str, Any]:
    frames = collect_frames(nusc, scene_limit, config.channel)
    scenes: dict[str, list[FrameRecord]] = {}
    for frame in frames:
        scenes.setdefault(frame.scene, []).append(frame)
    conditions = [("clean", 0)] + [
        (corruption, severity) for corruption in CORRUPTIONS for severity in range(1, 5)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    metrics: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    visualizations_written = 0

    for corruption, severity in tqdm(conditions, desc="Tracking conditions"):
        condition_rows: list[dict[str, object]] = []
        started = time.perf_counter()
        for scene_frames in scenes.values():
            tracker.reset()
            for batch_start in range(0, len(scene_frames), config.batch_size):
                batch = scene_frames[batch_start : batch_start + config.batch_size]
                if corruption == "clean":
                    images: list[Any] = [frame.image_path for frame in batch]
                    seeds = [0] * len(batch)
                else:
                    images, seeds = build_corrupted_batch(batch, corruption, severity, base_seed)
                predictions = detector.predict_batch(images)
                for frame, image, detections, seed in zip(
                    batch, images, predictions, seeds, strict=True
                ):
                    tracks = tracker.update(detections)
                    with Image.open(frame.image_path) as source:
                        width, height = source.size
                    truth = project_ground_truth(
                        nusc,
                        frame.sample_data_token,
                        width,
                        height,
                        config.min_visibility,
                        config.min_box_area,
                    )
                    matches = match_tracks(tracks, truth, config.iou_threshold)
                    condition_rows.extend(
                        _rows_for_frame(frame, tracks, truth, matches, corruption, severity, seed)
                    )
                    should_draw = (
                        corruption == "clean" and visualizations_written < config.max_visualizations
                    )
                    if should_draw and frame.frame % max(1, len(scene_frames) // 2) == 0:
                        draw_tracking(
                            image,
                            tracks,
                            truth,
                            matches,
                            output_dir
                            / "visualizations"
                            / f"{frame.scene}_{frame.frame:03d}_{corruption}_{severity}.jpg",
                            f"{corruption} severity {severity}",
                        )
                        visualizations_written += 1
        elapsed = time.perf_counter() - started
        key = f"{corruption}:{severity}"
        metrics[key] = identity_metrics(condition_rows)
        timings[key] = {
            "elapsed_seconds": elapsed,
            "frames_per_second": len(frames) / elapsed if elapsed else 0.0,
        }
        all_rows.extend(condition_rows)

    csv_path = output_dir / "tracking_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "detector": detector.backend_name,
        "tracker": tracker.backend_name,
        "scenes": len(scenes),
        "frames_per_condition": len(frames),
        "conditions": len(conditions),
        "keyframe_rate_hz": 2,
        "base_seed": base_seed,
        "config": asdict(config),
        "metrics": metrics,
        "timings": timings,
        "csv": str(csv_path.resolve()),
        "visualizations": visualizations_written,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_metrics(metrics, output_dir / "tracking_vs_severity.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ByteTrack on nuScenes CAM_FRONT keyframes."
    )
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scenes", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-visibility", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--min-box-area", type=float, default=400.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--track-buffer", type=int, default=6)
    parser.add_argument("--max-visualizations", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot.resolve()), verbose=False)
    config = TrackingConfig(
        min_visibility=args.min_visibility,
        min_box_area=args.min_box_area,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        batch_size=args.batch_size,
        track_buffer=args.track_buffer,
        max_visualizations=args.max_visualizations,
    )
    detector = UltralyticsYoloDetector(
        args.weights, args.device, args.confidence, args.image_size, args.batch_size
    )
    tracker = ByteTrackTracker(track_buffer=args.track_buffer)
    print(
        json.dumps(
            run_tracking_benchmark(
                nusc, detector, tracker, args.output_dir, config, args.scenes, args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
