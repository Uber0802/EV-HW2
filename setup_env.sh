#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="ev_hw2"

# ── 1. Create conda environment ───────────────────────────────────────────────
echo "=== [1/4] Creating conda environment: $ENV_NAME (Python 3.10) ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists — removing it first..."
    conda env remove -n "$ENV_NAME" -y
fi
conda create -n "$ENV_NAME" python=3.10 -y

# ── 2. Install CUDA 12.8 toolkit into conda env ───────────────────────────────
echo ""
echo "=== [2/4] Installing CUDA 12.8 toolkit into conda env ==="
conda install -n "$ENV_NAME" -c nvidia/label/cuda-12.8.0 cuda-toolkit -y

# ── 3. Install PyTorch + general dependencies ─────────────────────────────────
echo ""
echo "=== [3/4] Installing PyTorch 2.7.0 (CUDA 12.8) and dependencies ==="
conda run -n "$ENV_NAME" pip install \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

conda run -n "$ENV_NAME" pip install \
    plyfile==0.8.1 \
    tqdm \
    "numpy<2" \
    Pillow \
    "imageio==2.27.0" \
    imageio-ffmpeg \
    scipy \
    opencv-python \
    lpips \
    tensorboard

# ── 4. Build and install CUDA extensions ──────────────────────────────────────
echo ""
echo "=== [4/4] Building and installing CUDA extensions ==="

if [ ! -f "submodules/diff-gaussian-rasterization/setup.py" ]; then
    echo "  ERROR: submodules/diff-gaussian-rasterization not found."
    echo "  Run:  git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "submodules/simple-knn/setup.py" ]; then
    echo "  ERROR: submodules/simple-knn not found."
    echo "  Run:  git submodule update --init --recursive"
    exit 1
fi

echo "  Installing diff-gaussian-rasterization..."
TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    conda run -n "$ENV_NAME" pip install submodules/diff-gaussian-rasterization --no-build-isolation

echo "  Installing simple-knn..."
TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    conda run -n "$ENV_NAME" pip install submodules/simple-knn --no-build-isolation

# ── 5. Download D-NeRF dataset ─────────────────────────────────────────────────
echo "=== [5/5] Downloading D-NeRF dataset ==="
mkdir -p data/dnerf
FILE_ID="1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK"
wget -O data/dnerf/data.zip \
    "https://drive.google.com/uc?export=download&id=${FILE_ID}"
unzip data/dnerf/data.zip -d data/dnerf/
rm data/dnerf/data.zip

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
conda run -n "$ENV_NAME" python -c "
import torch
print(f'  torch                       : {torch.__version__}')
print(f'  CUDA available              : {torch.cuda.is_available()}')
import diff_gaussian_rasterization
print(f'  diff_gaussian_rasterization : OK')
import simple_knn
print(f'  simple_knn                  : OK')
import lpips
print(f'  lpips                       : OK')
"

