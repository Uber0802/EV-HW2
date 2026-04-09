#!/usr/bin/env bash
#
# Train / render / evaluate 4DGS on a D-NeRF style dataset (transforms_train.json).
#
# Optimization matches hustvl/4DGaussians merged config:
#   arguments/dnerf/dnerf_default.py
# Scene-specific HexPlane temporal resolution (e.g. bouncingballs) matches:
#   arguments/dnerf/bouncingballs.py  ->  resolution [64,64,64,75]
# HexPlane temporal resolution last dim is set in arguments/__init__.py (ModelHiddenParams
# kplanes_config). For bouncingballs, official repo uses 75 — ensure that matches your scene
# (see hustvl/4DGaussians arguments/dnerf/<scene>.py).
#
# Optional stability (defaults off, upstream parity): --stabilize_4dgs_loss 1 --grad_clip_norm_4dgs 2.0
# Deform/grid LR decay length uses position_lr_max_steps (same as hustvl gaussian_model).
#
# Usage:
#   chmod +x run_4dgs_dnerf.sh && ./run_4dgs_dnerf.sh
#   DATA_PATH=/path/to/dnerf/scene EXP_NAME=my_run ./run_4dgs_dnerf.sh
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# --- User-facing paths ---
: "${DATA_PATH:=data/dnerf/data/bouncingballs}"
: "${EXP_NAME:=4dgs_dnerf_bouncingballs_1}"


OUT_DIR="output/${EXP_NAME}"

echo "=========================================="
echo "4DGS (D-NeRF-aligned)  scene: ${DATA_PATH}"
echo "Output: ${OUT_DIR}"
echo "=========================================="

python train.py \
  -s "${DATA_PATH}" \
  -m "${OUT_DIR}" \
  --method 4dgs \
  --is_blender \
  --eval \
  --white_background \
  --iterations 20000 \
  --coarse_iterations 3000 \
  --deformation_lr_final 0.0000016 \
  --grid_lr_final 0.000016 \
  --position_lr_max_steps 20000 \
  --pruning_interval 8000 \
  --percent_dense 0.01

echo ""
echo "Rendering test/train views..."
python render.py \
  -m "${OUT_DIR}" \
  -s "${DATA_PATH}" \
  --method 4dgs \
  --is_blender \
  --eval \
  --white_background

echo ""
echo "Metrics (test/ours_* vs gt)..."
python metrics.py -m "${OUT_DIR}"

echo ""
echo "Done. PSNR/SSIM/LPIPS -> ${OUT_DIR}/results.json"
