#!/bin/bash
set -euo pipefail

if [[ -f /DATA/lym_data/miniforge3/etc/profile.d/conda.sh ]]; then
  source /DATA/lym_data/miniforge3/etc/profile.d/conda.sh
  conda activate drivestudio
fi

SCENE_ID="${1:-000}"
SCENE_ID="$(printf '%03d' "${SCENE_ID}")"

RESULTS_ROOT="/DATA/lym_data/omnire-results"
SCENE_DIR="${RESULTS_ROOT}/scene_${SCENE_ID}"
CHECKPOINT="${SCENE_DIR}/checkpoint_final.pth"
OUTPUT_DIR="${SCENE_DIR}/videos"

if [[ ! -e "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH=$(pwd)

for PRESET in normal slow fast; do
  echo "Rendering ego_speed_${PRESET}..."
  python tools/render_vehicle_speed_sim.py \
    --resume_from "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --preset "${PRESET}" \
    --cam_id 0
done

echo "Saved videos under: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/ego_speed_*.mp4
