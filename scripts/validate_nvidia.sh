#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
onnx_dir=${ONNX_DIR:-$project_root/outputs/milestone9/onnx}
engine_dir=${ENGINE_DIR:-$project_root/outputs/milestone9/tensorrt}
profile_dir=${PROFILE_DIR:-$project_root/outputs/milestone9/nsight}

for tool in nvidia-smi trtexec nsys; do
  if ! command -v "$tool" >/dev/null; then
    echo "NVIDIA validation blocked: required tool '$tool' is unavailable." >&2
    exit 2
  fi
done
nvidia-smi
trtexec --version
nsys --version

if [ ! -f "$onnx_dir/yolov8s.onnx" ] || [ ! -f "$onnx_dir/segformer_b0.onnx" ]; then
  "$project_root/.venv-linux/bin/export-onnx" --output-dir "$onnx_dir"
fi

"$project_root/scripts/build_tensorrt_engines.sh" "$onnx_dir" "$engine_dir"
"$project_root/scripts/profile_tensorrt.sh" "$engine_dir" "$profile_dir"

echo "NVIDIA validation complete. Do not report performance until trtexec and Nsight outputs exist."
