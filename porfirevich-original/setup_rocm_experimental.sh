#!/usr/bin/env bash
set -euo pipefail

archive_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${archive_dir}/.." && pwd)"
environment_dir="${workspace_dir}/.venv-rocm"
rocm_index="https://rocm.nightlies.amd.com/whl-multi-arch/"

if command -v uv >/dev/null 2>&1; then
  uv_path="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_path="${HOME}/.local/bin/uv"
else
  echo "Missing uv; install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

if [[ -e "${environment_dir}" ]]; then
  echo "Refusing to replace existing ${environment_dir}." >&2
  exit 1
fi

"${uv_path}" venv --python 3.12 "${environment_dir}"
python_path="${environment_dir}/bin/python"

"${uv_path}" pip install --python "${python_path}" \
  --index-url "${rocm_index}" \
  'amd-torch-device-gfx90c==2.12.0+rocm7.15.0a20260728'
"${uv_path}" pip install --python "${python_path}" \
  --index-url https://pypi.org/simple \
  'Cython==3.2.9' \
  'setuptools==81.0.0' \
  'wheel==0.47.0'
"${uv_path}" pip install --python "${python_path}" \
  --index-url https://pypi.org/simple \
  --no-build-isolation \
  'youtokentome==1.0.6'
"${uv_path}" pip install --python "${python_path}" \
  --index-url https://pypi.org/simple \
  'transformers==4.57.6' \
  'regex==2026.7.19'

echo "Experimental native gfx90c environment is ready at ${environment_dir}."
echo "Read ${archive_dir}/ROCM-EXPERIMENT.md before running it."
