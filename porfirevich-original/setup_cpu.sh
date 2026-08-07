#!/usr/bin/env bash
set -euo pipefail

archive_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${archive_dir}/.." && pwd)"
environment_dir="${workspace_dir}/.venv"
python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_include="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
yttm_include_flags=""
header_temp_dir=""

cleanup() {
  if [[ -n "${header_temp_dir}" && -d "${header_temp_dir}" ]]; then
    rm -rf -- "${header_temp_dir}"
  fi
}
trap cleanup EXIT

# YouTokenToMe 1.0.6 has no Python 3.14 wheel. If system headers are absent,
# extract Ubuntu's matching header package temporarily without requiring root.
if [[ ! -f "${python_include}/Python.h" ]]; then
  header_temp_dir="$(mktemp -d)"
  (
    cd "${header_temp_dir}"
    apt-get download "libpython${python_version}-dev"
    dpkg-deb -x libpython*.deb extracted
  )
  yttm_include_flags="-I${header_temp_dir}/extracted/usr/include -I${header_temp_dir}/extracted/usr/include/python${python_version}"
fi

python3 -m venv "${environment_dir}"
"${environment_dir}/bin/python" -m pip install \
  'Cython==3.2.9' \
  'setuptools==83.0.0' \
  'wheel==0.47.0'
env CFLAGS="${yttm_include_flags}" CXXFLAGS="${yttm_include_flags}" \
  "${environment_dir}/bin/python" -m pip install \
  --no-build-isolation \
  'youtokentome==1.0.6'
"${environment_dir}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.13.0'
"${environment_dir}/bin/python" -m pip install \
  'transformers==4.57.6' \
  'regex==2026.7.19'

echo "CPU environment is ready. Try:"
echo "${archive_dir}/run.sh 'Однажды вечером' --length 15 --seed 42"
