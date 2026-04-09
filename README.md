# EV-HW2 — Dynamic 3D Gaussian Splatting

This repository trains **deformable 3D Gaussians** and **4D Gaussians** on dynamic scenes (D-NeRF and related formats), with optional **SpeeDe3DGS**-style acceleration (temporal sensitivity sampling, score-based pruning) and **GroupFlow** on the deformable path.

For a **code-level map** (training dispatch, deformable vs 4DGS, SpeeDe, GroupFlow, rendering), see [`PROJECT_CODE_OVERVIEW.md`](PROJECT_CODE_OVERVIEW.md).

---

## Installation

### 1. Clone and submodules

```bash
git clone <this-repo-url> EV-HW2
cd EV-HW2
git submodule update --init --recursive
```

You need at least:

- `submodules/depth-diff-gaussian-rasterization` (differentiable rasterizer with depth; used by the project)
- `submodules/simple-knn`

### 2. Conda environment and Python dependencies

Create a conda env (example name `ev_hw2`, Python 3.10), then either:

**Option A — `setup_env.sh` (recommended)**  
Uncomment the “create env” / “CUDA toolkit” sections in [`setup_env.sh`](setup_env.sh) if you need them, set `TORCH_CUDA_ARCH_LIST` for your GPU (see comments in the script), then:

```bash
conda activate ev_hw2   # after creating the env
bash setup_env.sh
```

**Option B — manual**

```bash
conda create -n ev_hw2 python=3.10 -y
conda activate ev_hw2
# Optional: CUDA 12.8 toolkit in conda for nvcc (see setup_env.sh)
pip install -r requirements.txt
pip install submodules/depth-diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

[`requirements.txt`](requirements.txt) pins PyTorch **2.7.0+cu128** and runtime packages. For other CUDA versions, adjust the torch lines using [PyTorch Get Started](https://pytorch.org/get-started/locally/). `requirement.txt` is a thin wrapper that includes `requirements.txt`.

An alternative is [`environment.yaml`](environment.yaml) (`conda env create -f environment.yaml`), then build the two CUDA extensions as above.

---

## Dataset (D-NeRF — bouncing balls)

The **bouncing balls** scene from the D-NeRF synthetic data can be downloaded here:

[https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&st=t2vd1x4c&dl=0](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&st=t2vd1x4c&dl=0)

Unpack so that a scene contains `transforms_train.json` (Blender / D-NeRF layout). The example scripts expect:

```text
data/dnerf/data/bouncingballs/
```

Adjust `data_path` / `DATA_PATH` if you use another location.

---

## Running

### Deformable 3DGS (D-NeRF)

Script: [`run_d3dgs_dnerf.sh`](run_d3dgs_dnerf.sh)

```bash
chmod +x run_d3dgs_dnerf.sh
./run_d3dgs_dnerf.sh
```

This runs `train.py` with `--method deformable`, then `render.py`, then `metrics.py`. Edit `exp_name` and `data_path` inside the script as needed.

#### SpeeDe3DGS-style time-sensitive pruning (optional)

Uncomment the block in `run_d3dgs_dnerf.sh` (lines 27–32) to enable `--enable_speede_tricks` and related flags, for example:

- `--speede_prune_from_iter`, `--speede_prune_interval`
- `--speede_densify_prune_ratio`, `--speede_after_densify_prune_ratio`
- `--speede_use_tss --speede_use_vc` (temporal sensitivity sampling and view-count normalization)

#### GroupFlow (optional)

Uncomment the block in `run_d3dgs_dnerf.sh` (lines 33–36), e.g. `--gflow_flag`, `--gflow_iteration`, `--gflow_num`, `--gflow_opt`.

**Rendering with GroupFlow:** if you trained with GroupFlow, use `--gflow_flag` on `render.py` as well (see the commented example in `run_d3dgs_dnerf.sh` lines 44–46).

### 4D Gaussians — D-NeRF aligned (`hustvl/4DGaussians` style)

Script: [`run_4dgs_dnerf.sh`](run_4dgs_dnerf.sh)

```bash
chmod +x run_4dgs_dnerf.sh
./run_4dgs_dnerf.sh
```

Override paths if needed:

```bash
DATA_PATH=data/dnerf/data/bouncingballs EXP_NAME=my_4dgs_run ./run_4dgs_dnerf.sh
```

This calls `train.py --method 4dgs` with `--is_blender --eval --white_background`, then `render.py` and `metrics.py`. Scene-specific HexPlane settings for D-NeRF are documented in the script header and in [`PROJECT_CODE_OVERVIEW.md`](PROJECT_CODE_OVERVIEW.md).

---

## Source code references

This project builds on ideas and code patterns from:

| Reference | Description |
|-----------|-------------|
| [ingra14m/Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians?tab=readme-ov-file) | Deformable 3D Gaussians (CVPR 2024) — baseline deformable pipeline |
| [tuallen/speede3dgs](https://github.com/tuallen/speede3dgs) | SpeeDe3DGS — temporal pruning, TSS, motion grouping; optional features integrated here |
| [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) | 4D Gaussians — HexPlane + deformation; `--method 4dgs` path |

Please cite the original papers and repositories if you use this code in research.

---

## Reproducibility

Each training run writes `cfg_args` / `cfg_args.json` under the model output directory. Prefer fixed seeds and pinned `requirements.txt` when reporting numbers.
