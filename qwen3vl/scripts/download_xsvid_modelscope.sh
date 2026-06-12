#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./XS-VID}"

python -m pip install -U modelscope
modelscope download --dataset lanlanlanrr/XS-VID --local_dir "${OUT_DIR}"
mkdir -p "${OUT_DIR}/annotations" "${OUT_DIR}/images"
unzip -o "${OUT_DIR}/annotations.zip" -d "${OUT_DIR}/annotations"
find "${OUT_DIR}" -name 'videos_subset_*.zip' -exec unzip -o {} -d "${OUT_DIR}/images" \;
rm -f "${OUT_DIR}"/*.zip

echo "XS-VID downloaded to ${OUT_DIR}"
