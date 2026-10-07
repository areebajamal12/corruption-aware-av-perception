#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
linux_work="$project_root/work/linux"
uv_bin="$linux_work/bin/uv"
uv_cache="$linux_work/uv-cache"
uv_python="$linux_work/uv-python"

if [ "$(uname -s)" != "Linux" ]; then
  echo "setup_ubuntu.sh must run on Linux." >&2
  exit 1
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl git libgl1 libglib2.0-0

mkdir -p "$linux_work/bin" "$uv_cache" "$uv_python"
if [ ! -x "$uv_bin" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR="$linux_work/bin" UV_NO_MODIFY_PATH=1 sh
fi

UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" python install 3.11
UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" venv --python 3.11 --clear "$project_root/.venv-linux"
UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu \
  --python "$project_root/.venv-linux/bin/python"
UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" pip install -e "$project_root[dev,deployment]" \
  --python "$project_root/.venv-linux/bin/python"

echo "Ubuntu environment ready: $project_root/.venv-linux"
