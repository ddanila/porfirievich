#!/usr/bin/env bash
set -euo pipefail

archive_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${archive_dir}/.." && pwd)"
environment_dir="${workspace_dir}/.venv"
python_bin="${PYTHON:-python3}"
# Apple's system Python is too old. Homebrew may not be on the shell's PATH.
if [[ -z "${PYTHON:-}" && "$(uname -s)" == Darwin ]]; then
  for candidate in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if "${candidate}" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      python_bin="${candidate}"
      break
    fi
  done
fi
"${python_bin}" -c 'import sys; sys.exit("Python 3.10+ is required; set PYTHON to a newer interpreter.") if sys.version_info < (3, 10) else None'
python_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_include="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
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
  if [[ "$(uname -s)" != Linux ]] || ! command -v apt-get >/dev/null; then
    echo "Missing Python.h; install development headers for ${python_bin}." >&2
    exit 1
  fi
  header_temp_dir="$(mktemp -d)"
  (
    cd "${header_temp_dir}"
    apt-get download "libpython${python_version}-dev"
    dpkg-deb -x libpython*.deb extracted
  )
  yttm_include_flags="-I${header_temp_dir}/extracted/usr/include -I${header_temp_dir}/extracted/usr/include/python${python_version}"
fi

"${python_bin}" -m venv "${environment_dir}"
"${environment_dir}/bin/python" -m pip install \
  'Cython==3.2.9' \
  'setuptools==83.0.0' \
  'wheel==0.47.0'
env CFLAGS="${yttm_include_flags}" CXXFLAGS="${yttm_include_flags}" \
  "${environment_dir}/bin/python" -m pip install \
  --no-build-isolation \
  'youtokentome==1.0.6'
if [[ "$(uname -s)" == Darwin ]]; then
  # macOS wheels are published on PyPI, not the Linux CPU wheel index.
  "${environment_dir}/bin/python" -m pip install 'torch==2.13.0'
else
  "${environment_dir}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.13.0'
fi
"${environment_dir}/bin/python" -m pip install \
  'transformers==4.57.6' \
  'regex==2026.7.19'

echo "CPU environment is ready. Try:"
echo "${archive_dir}/run.sh 'Однажды вечером' --length 15 --seed 42"
