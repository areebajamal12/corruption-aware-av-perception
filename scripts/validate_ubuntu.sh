#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin="$project_root/.venv-linux/bin/python"
bin_dir="$project_root/.venv-linux/bin"

test "$(uname -s)" = "Linux"
test -x "$python_bin"

echo "OS validation"
uname -a
cat /etc/os-release
"$python_bin" --version

echo "Test suite"
"$bin_dir/ruff" check "$project_root/src" "$project_root/tests"
"$bin_dir/pytest" -q "$project_root/tests"

echo "CLI smoke tests"
for command in inspect-nuscenes evaluate-detection benchmark-corruptions \
  benchmark-segmentation benchmark-tracking evaluate-reliability; do
  "$bin_dir/$command" --help >/dev/null
  echo "validated: $command --help"
done

echo "Ubuntu validation complete."
