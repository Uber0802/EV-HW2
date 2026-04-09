# EV-HW2: Codebase Overview

This repository implements **dynamic 3D Gaussian Splatting** for **4D reconstruction** and **novel view synthesis**. Two training backends share one renderer and CLI: **Deformable 3DGS** (MLP deformation) and **4D Gaussians** (HexPlane spatiotemporal grid + MLP), selectable via `--method deformable` or `--method 4dgs`.

Optional **SpeeDe3DGS**-style acceleration (**temporal sensitivity sampling**, **view-count normalization**, **score-based pruning**) and **GroupFlow** (grouped motion after a chosen iteration) apply to the **deformable** path.

---

## Directory map (main code)

| Path | Role |
|------|------|
| `train.py` | Entry: dispatches `training_deformable` vs two-stage `scene_reconstruction` (4DGS), loss, densify/prune, SpeeDe pruning, GroupFlow |
| `render.py` | Load checkpoint, render test/train views |
| `metrics.py` | Evaluation on rendered folders |
| `arguments/__init__.py` | `ModelParams`, `PipelineParams`, `OptimizationParams`, `ModelHiddenParams` (4DGS architecture) |
| `scene/` | `Scene` loading, `GaussianModel`, `dataset_readers`, `deform_model.py`, `deformation.py` (4DGS network), `hexplane`, etc. |
| `gaussian_renderer/__init__.py` | Differentiable rasterization; supports delta vs absolute deformation |
| `utils/time_utils.py` | Deformable **DeformNetwork** |
| `utils/rigid_utils.py` | **GroupFlow** construction and per-frame steps |
| `submodules/` | `diff-gaussian-rasterization`, `simple-knn`, optional depth rasterizer |

---

## Method A: Deformable 3DGS (`--method deformable`)

- **Canonical Gaussians** live in `GaussianModel`; a small MLP (`DeformNetwork` in `utils/time_utils.py`) predicts **deltas** `d_xyz`, `d_rotation`, `d_scaling` from position + normalized time.
- **Warm-up**: for `iteration < warm_up`, deformation is disabled (zeros); then the deform network trains with the Gaussians.
- **Loss**: L1 + optional DSSIM (same form as 4DGS branch).
- **SpeeDe-style pruning** (optional): `enable_speede_tricks` → `_score_func_speede` accumulates per-Gaussian scores via rendered sum w.r.t. learnable `scores`; optional TSS noise on time; optional view-count normalization; periodic `_prune_speede` calls `gaussians.prune_gaussians`.
- **GroupFlow** (optional): after `gflow_iteration`, `do_group_flow` builds a grouped rigid model; `step_group_flow` can replace `deform.step` for training and evaluation.

---

## Method B: 4D Gaussians (`--method 4dgs`)

Matches the **hustvl/4DGaussians** schedule in spirit:

1. **Coarse stage**: pure static 3DGS (no deformation), `coarse_iterations` steps, fresh optimizer.
2. **Fine stage**: HexPlane + deformation MLP (`scene/deformation.py`); `DeformModel.step` returns **absolute** positions and activated scales/rotations; the renderer is called with `absolute_deform=True` so it does not add canonical + delta.
3. **Loss**: photometric + `deform.compute_regulation()` (HexPlane TV / time smoothness / L1 on planes as configured in `ModelHiddenParams`).

---

## Rendering convention (`gaussian_renderer/__init__.py`)

- **Deformable**: `means3D = pc.get_xyz + d_xyz` (and quaternion/scale composition with deltas), unless `is_6dof` (SE(3) path).
- **4DGS fine**: `absolute_deform=True` → `means3D = d_xyz` (already absolute), same for scales/rotations as implemented.

---

## Training pipeline (pseudocode)

```
procedure TRAIN:
  load dataset → Scene, GaussianModel, DeformModel(method)
  optionally: enable_speede_tricks preset (TSS + VC flags)

  if method == "4dgs":
    set AABB on deformation net from initial point cloud
    SCENE_RECONSTRUCTION(stage=coarse)   # no deform
    SCENE_RECONSTRUCTION(stage=fine)     # 4DGS + regulation
  else:
    TRAINING_DEFORMABLE()                # single stage

procedure SCENE_RECONSTRUCTION(stage):
  fresh optimizer; if fine and 4dgs: add deform params to unified Adam
  for iteration = 1 .. train_iter:
    update Gaussian LR; sample random camera; fid = time
    if stage == fine:
      (d_xyz, d_rot, d_scale) = deform.step(xyz, time, gaussians)  # 4DGS: absolute
    else:
      d_* = 0
    image = render(..., absolute_deform = (fine and 4dgs))
    loss = photometric
    if fine and 4dgs:
      loss += deform.compute_regulation()
    backward; densify/prune/opacity reset per schedule
    optimizer.step()

procedure TRAINING_DEFORMABLE:
  build gflow_model = None
  for iteration = 1 .. iterations:
    if iteration == gflow_iteration and gflow_flag:
      gflow_model = do_group_flow(...)
    sample camera; deform or GroupFlow or zero (warm_up)
    image = render(deltas)
    loss = photometric
    backward; densify_and_prune
    if speede_tricks and in prune window:
      _prune_speede(...)   # score-based prune
    optimizer steps (Gaussians, deform, optional GroupFlow)
```

---

## Config output

Each run writes `cfg_args` (pretty-printed `Namespace`) and `cfg_args.json` under the model path for reproducibility.

---

## Dependencies

See `requirement.txt` / project README for PyTorch, CUDA extensions (`diff-gaussian-rasterization`, etc.), and training data layout (`transforms_train.json` for Blender/D-NeRF style, or other loaders in `scene/dataset_readers.py`).
