"""Render a compact side-by-side perception demo from one nuScenes scene."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image

from av_perception.corruption.benchmark import derive_seed
from av_perception.corruption.engine import CORRUPTIONS, apply_corruption
from av_perception.detection.pipeline import collect_frames
from av_perception.detection.types import Detection
from av_perception.detection.yolo import UltralyticsYoloDetector
from av_perception.reliability.features import FEATURE_NAMES, image_quality_features
from av_perception.reliability.model import reliability_state
from av_perception.segmentation.segformer import CityscapesSegFormer
from av_perception.tracking.bytetrack import ByteTrackTracker
from av_perception.tracking.types import TrackedObject

STATE_COLORS = {
    "reliable": (60, 190, 80),
    "degraded": (30, 185, 245),
    "unsafe": (55, 55, 235),
}
TRACK_COLORS = {"vehicle": (255, 170, 30), "pedestrian": (220, 80, 220)}


def _batches(values: Sequence[Any], batch_size: int) -> list[Sequence[Any]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def runtime_feature_vector(
    image: np.ndarray,
    detections: list[Detection],
    road_mask: np.ndarray,
    tracks: list[TrackedObject],
    previous_track_ids: set[int],
) -> tuple[np.ndarray, set[int]]:
    """Build the exact observable feature vector used by the reliability estimator."""
    confidences = [detection.confidence for detection in detections]
    track_confidences = [track.confidence for track in tracks]
    track_ids = {track.track_id for track in tracks}
    continuity = (
        len(track_ids & previous_track_ids) / len(track_ids | previous_track_ids)
        if track_ids or previous_track_ids
        else 1.0
    )
    values: dict[str, float] = {
        **image_quality_features(image),
        "detection_count": float(len(detections)),
        "detection_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
        "detection_confidence_max": max(confidences, default=0.0),
        "vehicle_count": float(sum(item.class_name == "vehicle" for item in detections)),
        "pedestrian_count": float(sum(item.class_name == "pedestrian" for item in detections)),
        "predicted_road_fraction": float(road_mask.mean()),
        "track_count": float(len(tracks)),
        "track_confidence_mean": (
            float(np.mean(track_confidences)) if track_confidences else 0.0
        ),
        "track_continuity": continuity,
    }
    return np.asarray([[values[name] for name in FEATURE_NAMES]], dtype=float), track_ids


def _overlay_panel(
    image: np.ndarray,
    road_mask: np.ndarray,
    tracks: list[TrackedObject],
    title: str,
    score: float,
    width: int,
) -> np.ndarray:
    canvas = image.copy()
    tint = np.zeros_like(canvas)
    tint[road_mask] = (70, 180, 80)
    canvas = cv2.addWeighted(canvas, 1.0, tint, 0.34, 0.0)
    for track in tracks:
        color = TRACK_COLORS[track.class_name]
        box = track.bbox
        p1, p2 = (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2))
        cv2.rectangle(canvas, p1, p2, color, 3)
        label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
        cv2.putText(canvas, label, (p1[0], max(28, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    state = reliability_state(score)
    state_color = STATE_COLORS[state]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (20, 20, 20), -1)
    cv2.putText(canvas, title, (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
    text = f"{state.upper()}  {score:.2f}"
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    cv2.putText(
        canvas,
        text,
        (canvas.shape[1] - text_width - 18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        state_color,
        2,
    )
    height = round(canvas.shape[0] * width / canvas.shape[1])
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def _save_gif(frames: list[np.ndarray], destination: Path, fps: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    rgb[0].save(
        destination,
        save_all=True,
        append_images=rgb[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    for frame in rgb:
        frame.close()


def render_demo(args: argparse.Namespace) -> Path:
    """Run existing perception components and save an animated side-by-side GIF."""
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot.resolve()), verbose=False)
    frames = collect_frames(nusc, scene_limit=1, channel="CAM_FRONT")
    frames = frames[:: args.stride]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    clean_images = [cv2.imread(str(frame.image_path)) for frame in frames]
    if any(image is None for image in clean_images):
        raise FileNotFoundError("One or more CAM_FRONT frames could not be read.")
    corrupt_images = [
        apply_corruption(
            image,
            args.corruption,
            args.severity,
            derive_seed(args.seed, frame, args.corruption, args.severity),
        )
        for frame, image in zip(frames, clean_images, strict=True)
    ]

    detector = UltralyticsYoloDetector(
        weights=args.weights,
        device=args.device,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    segmenter = CityscapesSegFormer(model_name=args.segmentation_model, device=args.device)
    clean_detections = detector.predict_batch(clean_images)
    corrupt_detections = detector.predict_batch(corrupt_images)
    clean_masks: list[np.ndarray] = []
    corrupt_masks: list[np.ndarray] = []
    for batch in _batches(clean_images, args.batch_size):
        clean_masks.extend(segmenter.segment_batch(batch))
    for batch in _batches(corrupt_images, args.batch_size):
        corrupt_masks.extend(segmenter.segment_batch(batch))

    artifact = joblib.load(args.reliability_model)
    estimator = artifact["model"] if isinstance(artifact, dict) else artifact
    clean_tracker, corrupt_tracker = ByteTrackTracker(), ByteTrackTracker()
    previous_clean: set[int] = set()
    previous_corrupt: set[int] = set()
    rendered: list[np.ndarray] = []
    for index in range(len(frames)):
        clean_tracks = clean_tracker.update(clean_detections[index])
        corrupt_tracks = corrupt_tracker.update(corrupt_detections[index])
        clean_features, previous_clean = runtime_feature_vector(
            clean_images[index], clean_detections[index], clean_masks[index], clean_tracks, previous_clean
        )
        corrupt_features, previous_corrupt = runtime_feature_vector(
            corrupt_images[index],
            corrupt_detections[index],
            corrupt_masks[index],
            corrupt_tracks,
            previous_corrupt,
        )
        clean_score = float(estimator.predict_score(clean_features)[0])
        corrupt_score = float(estimator.predict_score(corrupt_features)[0])
        clean_panel = _overlay_panel(
            clean_images[index], clean_masks[index], clean_tracks, "CLEAN", clean_score, args.panel_width
        )
        corrupt_panel = _overlay_panel(
            corrupt_images[index],
            corrupt_masks[index],
            corrupt_tracks,
            f"{args.corruption.upper()}  S{args.severity}",
            corrupt_score,
            args.panel_width,
        )
        rendered.append(np.concatenate((clean_panel, corrupt_panel), axis=1))
    _save_gif(rendered, args.output, args.fps)
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a side-by-side nuScenes perception demo.")
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--output", type=Path, default=Path("assets/demo_noise_s4.gif"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--corruption", choices=sorted(CORRUPTIONS), default="noise")
    parser.add_argument("--severity", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument(
        "--segmentation-model",
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    )
    parser.add_argument(
        "--reliability-model",
        type=Path,
        default=Path("outputs/milestone7/full/reliability_model.joblib"),
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = render_demo(args)
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
