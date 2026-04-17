# ADVANCED_PROBLEM

This guide is for the advanced part of the homework.  
You will implement two core ideas from SpeeDe3DGS:

1. **Time-sensitive score for pruning + sampling (TSP + TSS)**  
2. **GroupFlow local rigid transform**

SpeeDe3DGS paper: [https://arxiv.org/pdf/2506.07917](https://arxiv.org/pdf/2506.07917)

The project already includes TODO anchors you should complete and then clean up.

---

## Part 1: Time-Sensitive Score (TSP + TSS)

### 1) Concept introduction

Dynamic 3DGS has many Gaussians, and not all are equally important over time.

- **TSP (Temporal Sensitivity Pruning)**: estimate each Gaussian's importance by how much it affects rendering over time, then prune low-score Gaussians.
- **TSS (Temporal Sensitivity Sampling)**: add small timestamp perturbation during score estimation to make pruning more robust to temporal instability (floaters/flicker).

In this codebase:
- score collection is done per-view in `train.py` inside `_score_func_speede(...)`
- score aggregation + pruning schedule is in `_prune_speede(...)`

### 2) Coarse code pipeline (pruning + sampling)

Read in this order:

1. `training_deformable(...)` in `train.py`  
   - checks whether to run SpeeDe pruning by iteration window and interval
2. `_prune_speede(...)` in `train.py`  
   - loops over all train cameras (time-aware accumulation)
   - calls `_score_func_speede(...)` per view
   - optional view-count normalization (VC)
   - calls `gaussians.prune_gaussians(...)`
3. `_score_func_speede(...)` in `train.py` (**your TODO target**)  
   - prepares time input and optional TSS noise
   - runs deformation (or GroupFlow)
   - renders with `scores=img_scores`
   - backward pass to accumulate per-Gaussian score gradients
4. `prune_gaussians(...)` in `scene/gaussian_model.py`  
   - removes lowest-score Gaussians by exact-count ranking

### 3) How score is computed (map to TODO)

TODO region: `train.py` in `_score_func_speede(...)` (the block between:
- `## TODO: Calculate scores for each Gaussian`
- `## TODO end of score calculation`)

Implementation meaning:

1. Build deformation state at this timestamp:
   - if GroupFlow active: `step_group_flow(...)`
   - else: `deform.step(x, time_input + tss_noise)`
2. Render with a learnable score proxy tensor:
   - pass `scores=img_scores` to `render(...)`
3. Backprop from image scalar:
   - `image.sum().backward()`
4. Accumulate per-view sensitivity:
   - preferred: `scores += img_scores.grad`
   - fallback: `scores[vis] += 1.0` when rasterizer has no score gradient

Interpretation:
- For each Gaussian `i`, score approximates view sensitivity and is summed across views.
- TSS changes the sampled time input (`time_input + noise`) to probe nearby time states.

### 4) Test setup for Part 1

Use only SpeeDe flags in `run_d3dgs_dnerf.sh`:

```bash
--enable_speede_tricks \
--speede_prune_from_iter 7000 \
--speede_prune_interval 4000 \
```

These correspond to lines `27-29` in your current script.

Run:

```bash
bash run_d3dgs_dnerf.sh
```

Expected checks:
- logs show SpeeDe trick enabled
- periodic pruning pass appears in training logs
- Gaussian count drops at pruning iterations

---

## Part 2: GroupFlow

### 1) Concept introduction

Per-Gaussian deformation is expensive.  
GroupFlow clusters Gaussians by motion and applies **shared group transforms**.

In this project, GroupFlow pipeline is:
- build groups and group trajectories
- interpolate group transform at time `t`
- map each Gaussian to group transform
- output per-point displacement `d_xyz`

### 2) Coarse code pipeline

Read in this order:

1. `training_deformable(...)` in `train.py`  
   - at `gflow_iteration`, calls `do_group_flow(...)`
2. `do_group_flow(...)` in `utils/rigid_utils.py`  
   - runs grouping (`grouping_stage_one(...)`)
   - initializes `GroupFlowModel_v2`
3. `step_group_flow(...)` in `utils/rigid_utils.py`  
   - dispatches runtime call to model `step_t(...)`
4. `GroupFlowModel_v2.step_t_func(...)`  
   - interpolates group-level transform `[R|T]` at timestamp `t`
5. `GroupFlowModel_v2.step_t(...)` (**your TODO target**)  
   - maps each point to nearest group
   - applies local rigid transform
   - returns `d_xyz`

### 3) GroupFlow local transform (map to TODO)

TODO region: `utils/rigid_utils.py` in `GroupFlowModel_v2.step_t(...)` (block between:
- `## TODO: Implement local rigid transform`
- `## TODO: End of local rigid transform`)

Implementation meaning:

1. Compute per-group transforms at `t`:
   - `gtransform = self.step_t_func(t, ...)` with shape `(Ng, 3, 4)`
2. Map each Gaussian to one group:
   - `point_labels = self._refresh_point_labels(x)`
3. Gather per-point transform:
   - `ptransform = gtransform[point_labels]`
4. Apply local rigid motion around node center:
   - `x' = R @ (x - node) + node + T`
   - `d_xyz = x' - x`

This returns translation offset for renderer/deformation path.

### 4) Final test setup (SpeeDe + GroupFlow)

Use full flag set in `run_d3dgs_dnerf.sh`:

```bash
--enable_speede_tricks \
--speede_prune_from_iter 7000 \
--speede_prune_interval 4000 \
--gflow_flag \
--gflow_iteration 15000 \
--gflow_num 2048 \
```

These correspond to lines `27-32` in your current script.

Run:

```bash
bash run_d3dgs_dnerf.sh
```

Expected checks:
- SpeeDe pruning logs appear before/around prune intervals
- at `gflow_iteration`, GroupFlow build log appears
- later iterations use GroupFlow branch for deformation

---

## Submission checklist

- Complete both TODO sections:
  - `train.py` `_score_func_speede(...)`
  - `utils/rigid_utils.py` `GroupFlowModel_v2.step_t(...)`
- Verify training runs with:
  - SpeeDe-only test (lines `27-29`)
  - SpeeDe + GroupFlow test (lines `27-32`)
- Remove TODO markers/comments before final submission.

