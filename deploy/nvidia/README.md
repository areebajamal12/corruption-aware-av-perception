# NVIDIA deployment and validation

No NVIDIA execution result is claimed in this repository unless it was produced on an actual
NVIDIA GPU. The local Apple Silicon host and Ubuntu ARM64 VM cannot validate CUDA or TensorRT.

## 1. Export and validate ONNX anywhere

```bash
uv pip install -e '.[deployment]'
export-onnx --output-dir outputs/milestone9/onnx
```

This exports YOLOv8s (`images`, dynamic batch, 640x640) and SegFormer-B0 (`pixel_values`, dynamic
batch/height/width), runs the ONNX checker, and executes zero-input inference with ONNX Runtime.
The generated models are intentionally ignored by Git.

## 2. NVIDIA prerequisites

Use Ubuntu with an NVIDIA driver, CUDA-capable GPU, TensorRT's non-pip distribution (which includes
`trtexec`), and Nsight Systems (`nsys`). Confirm all three before running anything:

```bash
nvidia-smi
trtexec --version
nsys --version
```

TensorRT engines are hardware/software specific. Build them on the deployment GPU; do not commit or
move `.engine` files between unrelated GPU/TensorRT versions.

## 3. Build FP16 TensorRT engines

```bash
./scripts/build_tensorrt_engines.sh \
  outputs/milestone9/onnx outputs/milestone9/tensorrt
```

The script uses explicit min/opt/max shapes for batch 1–8, detailed profiling verbosity, and saves
separate detector and segmenter engines. Preserve the complete `trtexec` output; it is the source of
latency, throughput, memory, and layer/tactic evidence.

## 4. Profile with Nsight Systems

```bash
./scripts/profile_tensorrt.sh \
  outputs/milestone9/tensorrt outputs/milestone9/nsight
```

This profiles 200 warmed-up CUDA-graph inference iterations with host/device transfers disabled so
kernel scheduling and engine execution can be inspected. End-to-end camera preprocessing and
postprocessing must be profiled separately before making real-time pipeline claims.

`./scripts/validate_nvidia.sh` runs the prerequisite checks, export (if needed), engine builds, and
profiles as one fail-fast workflow.

## Free GPU execution

As of October 2026, Kaggle's official documentation says notebooks can use one NVIDIA Tesla P100
for free, subject to weekly quota, queueing, and availability. Google Colab also has a free tier,
but its official FAQ states that resources and GPU availability are not guaranteed. Kaggle is the
preferred free attempt because its documentation is explicit about the P100 and quota.

Starting either service requires the user's account, agreement to its terms, and manually enabling
the GPU accelerator. Do not select paid upgrades, attach billing, or use Google Cloud export. Once
a free GPU session is visible in `nvidia-smi`, upload/clone this repository and run the steps above.

## Evidence checklist

Record these without inventing missing values:

- GPU name, driver, CUDA, TensorRT, and Nsight versions
- Git commit and ONNX SHA-256 hashes
- TensorRT build success and precision/tactic warnings
- batch-1 latency percentiles, throughput, and GPU memory from `trtexec`
- `.nsys-rep` files and a short description of the dominant kernels/gaps
- detector/segmenter numerical comparison against ONNX Runtime on the same inputs
- full end-to-end timing including preprocessing and postprocessing
