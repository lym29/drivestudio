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
OUTPUT="${SCENE_DIR}/videos/novel_orbit_background.mp4"

if [[ ! -e "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${SCENE_DIR}/videos"

export PYTHONPATH=$(pwd)

python tools/render_novel_background.py \
  --resume_from "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --traj_type orbit_pullback \
  --height_m 0.2 \
  --height_axis_sign -1 \
  --elevate_frames 15 \
  --spin_frames 360 \
  --exit_transition_frames 20 \
  --fps 24

echo "Saved: ${OUTPUT}"
