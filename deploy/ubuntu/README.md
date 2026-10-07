# Ubuntu validation

Project 1 supports Ubuntu on CPU and NVIDIA systems. The setup is intentionally local and does not
require cloud infrastructure.

```bash
./scripts/setup_ubuntu.sh
./scripts/validate_ubuntu.sh
```

`setup_ubuntu.sh` installs required Ubuntu packages, a project-local `uv`, Python 3.11, and all
Python dependencies into `.venv-linux`. It never reuses the macOS `.venv`. `validate_ubuntu.sh`
confirms the Linux kernel and distribution, runs Ruff and pytest, then invokes every project CLI.

The final validation was performed in a free local Ubuntu 24.04.4 LTS ARM64 Lima VM on an Apple
Silicon host: 39 tests and all seven CLI entry points passed. This validates Linux setup, imports,
tests, and CLI entry points. It does not validate
CUDA, TensorRT, or NVIDIA performance because the local VM has no NVIDIA GPU.
