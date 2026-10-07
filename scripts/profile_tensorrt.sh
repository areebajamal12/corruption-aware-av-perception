#!/bin/sh
set -eu

engine_dir=${1:-outputs/milestone9/tensorrt}
profile_dir=${2:-outputs/milestone9/nsight}
mkdir -p "$profile_dir"

command -v nvidia-smi >/dev/null
command -v trtexec >/dev/null
command -v nsys >/dev/null

for model in yolov8s_fp16 segformer_b0_fp16; do
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
    --output="$profile_dir/$model" \
    trtexec --loadEngine="$engine_dir/$model.engine" \
      --profilingVerbosity=detailed --warmUp=1000 --duration=0 --iterations=200 \
      --useCudaGraph --noDataTransfers
done

echo "Nsight Systems reports written to $profile_dir"
