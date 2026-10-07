#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv_bin="$project_root/work/bin/uv"
uv_cache="$project_root/work/uv-cache"
uv_python="$project_root/work/uv-python"

mkdir -p "$project_root/work/bin" "$uv_cache" "$uv_python"

if [ ! -x "$uv_bin" ]; then
  echo "Installing uv inside the project..."
  curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR="$project_root/work/bin" UV_NO_MODIFY_PATH=1 sh
fi

echo "Installing project-local Python 3.11..."
UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" python install 3.11

echo "Creating .venv and installing project dependencies..."
UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" venv --python 3.11 "$project_root/.venv"
UV_CACHE_DIR="$uv_cache" UV_PYTHON_INSTALL_DIR="$uv_python" \
  "$uv_bin" pip install -e "$project_root[dev]" --python "$project_root/.venv/bin/python"

echo "Environment ready. Activate it with: source .venv/bin/activate"
