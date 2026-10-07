"""Export detection and segmentation models to portable ONNX graphs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def validate_onnx(path: Path, input_name: str, shape: tuple[int, ...]) -> dict[str, object]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {input_name: np.zeros(shape, dtype=np.float32)})
    external_files = sorted(path.parent.glob(f"{path.name}.data*"))
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size + sum(item.stat().st_size for item in external_files),
        "external_data_files": [item.name for item in external_files],
        "input": input_name,
        "input_shape": list(shape),
        "output_names": [value.name for value in session.get_outputs()],
        "output_shapes": [list(value.shape) for value in outputs],
    }


def export_yolo(weights: str, destination: Path, image_size: int) -> dict[str, object]:
    from ultralytics import YOLO

    exported = Path(
        YOLO(weights).export(
            format="onnx",
            imgsz=image_size,
            dynamic=True,
            opset=17,
            simplify=False,
            nms=False,
            device="cpu",
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != destination.resolve():
        shutil.move(str(exported), destination)
    return validate_onnx(destination, "images", (1, 3, image_size, image_size))


class _SegmentationLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export_segformer(model_name: str, destination: Path, image_size: int) -> dict[str, object]:
    from transformers import AutoModelForSemanticSegmentation

    wrapper = _SegmentationLogits(AutoModelForSemanticSegmentation.from_pretrained(model_name))
    wrapper.eval()
    example = torch.zeros(1, 3, image_size, image_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        destination,
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        dynamo=True,
        dynamic_shapes=({0: "batch", 2: "height", 3: "width"},),
    )
    return validate_onnx(destination, "pixel_values", tuple(example.shape))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and validate Project 1 ONNX models.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument(
        "--segmentation-model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )
    parser.add_argument("--detection-size", type=int, default=640)
    parser.add_argument("--segmentation-size", type=int, default=512)
    parser.add_argument("--model", choices=("all", "detection", "segmentation"), default="all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results: dict[str, object] = {}
    if args.model in {"all", "detection"}:
        results["detection"] = export_yolo(
            args.weights, args.output_dir / "yolov8s.onnx", args.detection_size
        )
    if args.model in {"all", "segmentation"}:
        results["segmentation"] = export_segformer(
            args.segmentation_model, args.output_dir / "segformer_b0.onnx", args.segmentation_size
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "onnx_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
