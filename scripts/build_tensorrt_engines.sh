#!/bin/sh
set -eu

onnx_dir=${1:-outputs/milestone9/onnx}
engine_dir=${2:-outputs/milestone9/tensorrt}
mkdir -p "$engine_dir"

command -v nvidia-smi >/dev/null
command -v trtexec >/dev/null
nvidia-smi

trtexec --onnx="$onnx_dir/yolov8s.onnx" \
  --saveEngine="$engine_dir/yolov8s_fp16.engine" --fp16 \
  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 \
  --maxShapes=images:8x3x640x640 --profilingVerbosity=detailed

trtexec --onnx="$onnx_dir/segformer_b0.onnx" \
  --saveEngine="$engine_dir/segformer_b0_fp16.engine" --fp16 \
  --minShapes=pixel_values:1x3x512x512 --optShapes=pixel_values:1x3x512x512 \
  --maxShapes=pixel_values:8x3x512x512 --profilingVerbosity=detailed

echo "TensorRT engines built in $engine_dir"
