"""Run object detection and evaluation over nuScenes CAM_FRONT keyframes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
from tqdm import tqdm

from av_perception.detection.base import ObjectDetector
from av_perception.detection.evaluation import match_frame, summarize_metrics
from av_perception.detection.ground_truth import project_ground_truth
from av_perception.detection.types import Detection, DetectionMatch, GroundTruth
from av_perception.detection.yolo import UltralyticsYoloDetector

CSV_FIELDS = [
    "scene",
    "frame",
    "corruption",
    "severity",
    "seed",
    "sample_token",
    "image_path",
    "record_type",
    "class",
    "source_class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "iou",
    "matched",
    "matched_annotation_token",
    "visibility",
]


@dataclass(frozen=True)
class EvaluationConfig:
    channel: str = "CAM_FRONT"
    min_visibility: int = 2
    min_box_area: float = 400.0
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.25
    image_size: int = 640
    batch_size: int = 8
    max_visualizations: int = 12


@dataclass(frozen=True)
class FrameRecord:
    scene: str
    frame: int
    sample_token: str
    sample_data_token: str
    image_path: Path
    width: int
    height: int


def collect_frames(nusc: Any, scene_limit: int | None, channel: str) -> list[FrameRecord]:
    """Collect keyframes in scene/time order."""
    records: list[FrameRecord] = []
    scenes = nusc.scene[:scene_limit] if scene_limit is not None else nusc.scene
    for scene in scenes:
        token = scene["first_sample_token"]
        frame_index = 0
        while token:
            sample = nusc.get("sample", token)
            sample_data_token = sample["data"][channel]
            sample_data = nusc.get("sample_data", sample_data_token)
            image_path = Path(nusc.get_sample_data_path(sample_data_token))
            records.append(
                FrameRecord(
                    scene=scene["name"],
                    frame=frame_index,
                    sample_token=token,
                    sample_data_token=sample_data_token,
                    image_path=image_path,
                    width=int(sample_data["width"]),
                    height=int(sample_data["height"]),
                )
            )
            token = sample["next"]
            frame_index += 1
    return records


def _base_row(
    frame: FrameRecord, corruption: str = "clean", severity: int = 0, seed: int = 0
) -> dict[str, object]:
    return {
        "scene": frame.scene,
        "frame": frame.frame,
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "sample_token": frame.sample_token,
        "image_path": str(frame.image_path),
    }


def make_result_rows(
    frame: FrameRecord,
    detections: list[Detection],
    ground_truth: list[GroundTruth],
    matches: list[DetectionMatch],
    corruption: str = "clean",
    severity: int = 0,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Create reusable long-form rows for predictions and filtered ground truth."""
    rows: list[dict[str, object]] = []
    matches_by_detection = {match.detection_index: match for match in matches}
    matched_gt_indices = {
        match.ground_truth_index for match in matches if match.ground_truth_index is not None
    }
    for detection_index, detection in enumerate(detections):
        match = matches_by_detection[detection_index]
        truth = (
            ground_truth[match.ground_truth_index] if match.ground_truth_index is not None else None
        )
        rows.append(
            {
                **_base_row(frame, corruption, severity, seed),
                "record_type": "detection",
                "class": detection.class_name,
                "source_class": detection.source_class,
                "confidence": detection.confidence,
                **asdict(detection.bbox),
                "iou": match.iou,
                "matched": match.matched,
                "matched_annotation_token": truth.annotation_token if truth else "",
                "visibility": truth.visibility if truth else "",
            }
        )
    for gt_index, truth in enumerate(ground_truth):
        rows.append(
            {
                **_base_row(frame, corruption, severity, seed),
                "record_type": "ground_truth",
                "class": truth.class_name,
                "source_class": truth.source_category,
                "confidence": "",
                **asdict(truth.bbox),
                "iou": "",
                "matched": gt_index in matched_gt_indices,
                "matched_annotation_token": truth.annotation_token,
                "visibility": truth.visibility,
            }
        )
    return rows


def draw_comparison(
    frame: FrameRecord,
    detections: list[Detection],
    ground_truth: list[GroundTruth],
    matches: list[DetectionMatch],
    destination: Path,
) -> None:
    """Save GT (green) and predictions (blue matched, red unmatched)."""
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise FileNotFoundError(frame.image_path)
    match_lookup = {match.detection_index: match for match in matches}
    for truth in ground_truth:
        box = truth.bbox
        cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 220, 0), 2)
        if box.area >= 2_000:
            cv2.putText(
                image,
                f"GT {truth.class_name}",
                (int(box.x1), max(15, int(box.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
    for index, detection in enumerate(detections):
        match = match_lookup[index]
        color = (255, 120, 0) if match.matched else (0, 0, 255)
        box = detection.bbox
        cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        if box.area >= 1_000:
            cv2.putText(
                image,
                f"P {detection.class_name} {detection.confidence:.2f} IoU {match.iou:.2f}",
                (int(box.x1), min(image.shape[0] - 5, int(box.y2) + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        image,
        "GT: green | matched prediction: blue | unmatched prediction: red",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Failed to write visualization: {destination}")


def evaluate_dataset(
    nusc: Any,
    detector: ObjectDetector,
    output_dir: Path,
    config: EvaluationConfig,
    scene_limit: int | None,
) -> dict[str, object]:
    frames = collect_frames(nusc, scene_limit, config.channel)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    visualization_stride = max(1, len(frames) // max(config.max_visualizations, 1))
    visualizations_written = 0
    started = time.perf_counter()

    for batch_start in tqdm(range(0, len(frames), config.batch_size), desc="Detecting"):
        batch = frames[batch_start : batch_start + config.batch_size]
        predictions = detector.predict_batch([frame.image_path for frame in batch])
        if len(predictions) != len(batch):
            raise RuntimeError("Detector returned a different number of results than input frames.")
        for offset, (frame, detections) in enumerate(zip(batch, predictions, strict=True)):
            with Image.open(frame.image_path) as image:
                width, height = image.size
            ground_truth = project_ground_truth(
                nusc,
                frame.sample_data_token,
                width,
                height,
                min_visibility=config.min_visibility,
                min_box_area=config.min_box_area,
            )
            matches = match_frame(detections, ground_truth, config.iou_threshold)
            rows.extend(make_result_rows(frame, detections, ground_truth, matches))
            global_index = batch_start + offset
            if (
                visualizations_written < config.max_visualizations
                and global_index % visualization_stride == 0
            ):
                draw_comparison(
                    frame,
                    detections,
                    ground_truth,
                    matches,
                    output_dir / "visualizations" / f"{frame.scene}_{frame.frame:03d}.jpg",
                )
                visualizations_written += 1

    elapsed = time.perf_counter() - started
    csv_path = output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_metrics(rows, config.iou_threshold)
    summary.update(
        {
            "backend": detector.backend_name,
            "channel": config.channel,
            "scenes": len({frame.scene for frame in frames}),
            "frames": len(frames),
            "elapsed_seconds": elapsed,
            "frames_per_second": len(frames) / elapsed if elapsed else 0.0,
            "config": asdict(config),
            "csv": str(csv_path.resolve()),
            "visualizations": visualizations_written,
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8s on nuScenes CAM_FRONT.")
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
    parser.add_argument("--max-visualizations", type=int, default=12)
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
        max_visualizations=args.max_visualizations,
    )
    detector = UltralyticsYoloDetector(
        weights=args.weights,
        device=args.device,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    summary = evaluate_dataset(nusc, detector, args.output_dir, config, args.scenes)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
