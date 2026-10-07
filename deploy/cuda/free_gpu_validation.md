# Free GPU handoff

1. Open a Kaggle Notebook under an account with unused free GPU quota.
2. In **Settings → Accelerator**, select the free GPU option. Do not choose any paid/cloud upgrade.
3. Verify the allocation with `!nvidia-smi` before installing anything.
4. Clone the private repository using a short-lived, narrowly scoped credential entered by the user,
   or upload a source archive that excludes data, outputs, environments, weights, and credentials.
5. Install the deployment dependencies and TensorRT/Nsight tooling available in the notebook image.
6. Run `export-onnx`, `build_tensorrt_engines.sh`, and `profile_tensorrt.sh` as supported.
7. Download only the JSON/log/profile evidence; never commit TensorRT engines or credentials.

If Kaggle has no free accelerator available, disconnect without enabling billing and try again later.
Colab free is a fallback, but availability is explicitly not guaranteed.
