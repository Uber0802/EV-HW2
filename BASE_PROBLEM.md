# BASE_PROBLEM

This guide is for the **base homework** of dynamic 4D reconstruction in this project.  
You will complete core implementations for:

1. **Deformable 3DGS (d3dgs)** time-conditioned deformation
2. **4DGS** HexPlane-based dynamic deformation
3. Deformation-to-renderer integration
4. Final training loss design

Papers:
- Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction: [https://arxiv.org/pdf/2309.13101](https://arxiv.org/pdf/2309.13101)
- 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering: [https://arxiv.org/pdf/2310.08528](https://arxiv.org/pdf/2310.08528)

---

## Learning goals

By finishing this homework, you should understand:

- how time embedding `gamma(t)` conditions deformation in d3dgs
- how 4DGS uses HexPlane multi-resolution features over `(x, y, z, t)`
- why d3dgs uses **delta deformation** while 4DGS fine stage uses **absolute deformation**
- how photometric + regularization losses are combined in dynamic Gaussian training

---

## Part 0: Read code in this order

Before writing code, read these files top-down:

1. `train.py`  
   - training loop, deformation call, render call, loss composition
2. `scene/deform_model.py`  
   - unified deform interface for `deformable` and `4dgs`
3. `gaussian_renderer/__init__.py`  
   - how deformed attributes are fused with canonical Gaussians
4. `utils/time_utils.py`  
   - d3dgs positional/time embedding and DeformNetwork
5. `scene/hexplane.py` and `scene/deformation.py`  
   - 4DGS multi-resolution HexPlane feature query and deformation heads
6. `scene/regulation.py`  
   - plane smoothness operator used by 4DGS regulation

---

## Part 1: d3dgs time embedding `gamma(t)`

### Concept

In d3dgs, Gaussian motion is predicted by an MLP from:
- embedded 3D position `gamma(x)`
- embedded time `gamma(t)`

The embedding is sinusoidal positional encoding (multi-frequency `sin/cos`), not a raw scalar timestamp.

### Key reference code (already implemented)

- `utils/time_utils.py`
  - `get_embedder(...)`
  - `Embedder.create_embedding_fn(...)`
  - `DeformNetwork.forward(...)` with:
    - `t_emb = self.embed_time_fn(t)`
    - `x_emb = self.embed_fn(x)`

### What to check while coding

- input `t` is normalized to `[0,1]` by camera `fid`
- embedding dimensions are consistent with MLP input
- forward output shape matches `(N, 3)`, `(N, 4)`, `(N, 3)`

---

## Part 2: 4DGS HexPlane multi-resolution + normalization

### Concept

4DGS does not use one single grid. It uses **multi-resolution HexPlane features**:

- each level stores factorized planes over combinations of coordinates
- each query point `(x, y, z, t)` samples plane features
- features are multiplied within one level, then combined across levels
- low-res levels capture global dynamics, high-res levels capture local detail

### Key reference code (already implemented)

- `scene/hexplane.py`
  - `interpolate_ms_features(...)`
  - `HexPlaneField.__init__(...)` (multi-resolution construction)
  - `normalize_aabb(...)`
  - `HexPlaneField.get_density(...)` (AABB + time normalization and interpolation)
- `scene/deformation.py`
  - `Deformation.query_time(...)` and `forward_dynamic(...)`

### What to check while coding

- xyz is normalized by AABB to `[-1, 1]`
- time is remapped from `[0,1]` to `[-1,1]` before `grid_sample`
- scale-level feature fusion follows design (concat or sum)

---

## Part 3: TODO implementation map

Complete these `## TODO` blocks in the base assignment:

1. `train.py`
   - `## TODO: Implement L1 loss and SSIM loss`
   - `## TODO: End of 4DGS HexPlane TV regularisation`

2. `gaussian_renderer/__init__.py`
   - `## TODO: Combine deformation and canonical Gaussian positions`
   - `## TODO: End of combining deformation and canonical Gaussian positions`

3. `scene/deform_model.py`
   - `## TODO: Implement plane smoothness and l1 regulation`
   - `## TODO: End of plane smoothness and l1 regulation`

4. `scene/hexplane.py`
   - `## TODO: Implement multi-scale feature interpolation`
   - `## TODO: End of multi-scale feature interpolation`

---

## Part 4: Algorithm pipeline (end-to-end)

### A) Deformable 3DGS (`--method deformable`)

1. sample a training camera and its time `fid`
2. build `time_input` for all Gaussians
3. run `deform.step(xyz, time_input)` to get deltas
4. renderer composes canonical + delta attributes
5. compute photometric loss
6. backward + optimizer + densify/prune schedule

### B) 4DGS (`--method 4dgs`)

1. coarse stage: static Gaussian training
2. fine stage:
   - query HexPlane with `(x,y,z,t)`
   - predict absolute dynamic attributes
   - render with `absolute_deform=True`
3. loss = photometric + HexPlane regulation
4. backward + optimizer + densify/prune schedule

---

## Part 5: Final loss design to implement/check

### Photometric term (both methods)

Use:

`loss_photo = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)`

in `train.py` training loop.

### 4DGS regularization term (fine stage only)

In `scene/deform_model.py -> compute_regulation()`:

- spatial plane smoothness on spatial planes
- temporal smoothness on time-including planes
- L1 penalty on temporal planes

Final:

`loss = loss_photo + loss_reg_4dgs` (only for 4DGS fine stage)

---

## Part 6: Test and run

Use the provided scripts:

### d3dgs run

```bash
bash run_d3dgs_dnerf.sh
```

### 4dgs run

```bash
bash run_4dgs_dnerf.sh
```

Recommended checks:

- training starts without shape/type errors
- loss is finite (no persistent NaN/Inf)
- rendering and metrics stages complete
- output folders and `results.json` are produced

---

## Submission checklist

- complete all base `## TODO` blocks listed above
- run both scripts at least once:
  - `run_d3dgs_dnerf.sh`
  - `run_4dgs_dnerf.sh`
- verify training/render/metrics complete and produce outputs
- keep code comments clear and concise for report/demo
