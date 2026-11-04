#!/usr/bin/env bash
set -euo pipefail

archive_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${archive_dir}/model/m_checkpoint-3365357"
base_url="https://s3.us-west-2.amazonaws.com/models.dobro.ai/gpt2/ru/pelevin/m_checkpoint-3365357"

mkdir -p "${model_dir}"

download() {
  local filename="$1"
  local expected_size="$2"
  local destination="${model_dir}/${filename}"
  local actual_size=0

  if [[ -f "${destination}" ]]; then
    actual_size="$(wc -c < "${destination}" | tr -d '[:space:]')"
  fi
  if [[ "${actual_size}" == "${expected_size}" ]]; then
    echo "Already complete: ${filename}"
    return
  fi

  curl \
    --fail \
    --location \
    --retry 10 \
    --retry-all-errors \
    --continue-at - \
    --output "${destination}" \
    "${base_url}/${filename}"
}

download config.json 674
download encoder.model 724643
download pytorch_model.bin 1725858844
download step.txt 7
download training_args.bin 1420

cd "${archive_dir}"
if command -v sha256sum >/dev/null; then
  sha256sum --check SHA256SUMS
else
  shasum -a 256 --check SHA256SUMS
fi
