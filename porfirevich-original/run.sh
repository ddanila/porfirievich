#!/usr/bin/env bash
set -euo pipefail

archive_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${archive_dir}/.." && pwd)"
environment_name="${PORFIRIEVICH_VENV:-.venv}"
if [[ "${environment_name}" == /* ]]; then
  environment_dir="${environment_name}"
else
  environment_dir="${workspace_dir}/${environment_name}"
fi
python_path="${environment_dir}/bin/python"

if [[ ! -x "${python_path}" ]]; then
  echo "Missing ${python_path}; create the environment described in requirements-cpu.txt." >&2
  exit 1
fi

exec env PYTHONDONTWRITEBYTECODE=1 \
  "${python_path}" "${archive_dir}/run_cpu.py" "$@"
