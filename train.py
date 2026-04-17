"""
Unified training script for Dynamic 3D Gaussian Splatting.

Supports two methods via --method flag:
  deformable  –  Deformable 3D Gaussians (MLP deformation field)
  4dgs        –  4D Gaussians (HexPlane 4D spatiotemporal feature grid)

4DGS training exactly matches hustvl/4DGaussians:
  - Stage 1 (coarse): pure 3DGS, opt.coarse_iterations steps, fresh optimizer
  - Stage 2 (fine):   4DGS with HexPlane, opt.iterations steps, fresh unified optimizer
  - Unified optimizer: Gaussian params + deformation params share one Adam
  - LR updated at START of each iteration (before forward pass)
  - NaN loss detection: skip optimizer step to avoid weight corruption
  - Stage-dependent densification thresholds
"""

import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.deform_model import DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from pprint import pformat
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.rigid_utils import do_group_flow, step_group_flow

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ── Unified-optimizer helper (4DGS only) ─────────────────────────────────────

def _add_deform_to_optimizer(gaussians, deform, opt):
    """Add deformation network params to gaussians.optimizer (fine stage only).

    This replicates the original 4DGaussians design where Gaussian params and
    deformation params are trained by a single Adam optimizer, ensuring
    consistent LR scheduling and Adam momentum across the whole model.
    """
    spatial_lr = gaussians.spatial_lr_scale
    deformation_net = deform.deform.deformation_net

    gaussians.optimizer.add_param_group({
        'params': list(deformation_net.get_mlp_parameters()),
        'lr': opt.deformation_lr_init * spatial_lr,
        'name': 'deform_mlp',
    })
    gaussians.optimizer.add_param_group({
        'params': list(deformation_net.get_grid_parameters()),
        'lr': opt.grid_lr_init * spatial_lr,
        'name': 'deform_grid',
    })


def _set_unified_deform_learning_rates(gaussians, deform, iteration: int):
    """Apply DeformModel LR schedulers to deform_mlp / deform_grid in gaussians.optimizer.

    The fine-stage optimizer is unified (Gaussian + HexPlane params in one Adam).
    gaussians.update_learning_rate only updates the xyz group; deform.*_scheduler_args
    must be written into the unified optimizer each step (deform.optimizer is unused).
    """
    for param_group in gaussians.optimizer.param_groups:
        if param_group["name"] == "deform_mlp":
            param_group["lr"] = deform.mlp_scheduler_args(iteration)
        elif param_group["name"] == "deform_grid":
            param_group["lr"] = deform.grid_scheduler_args(iteration)


def _apply_speede_tricks_preset(opt):
    """Enable all three SpeeDe3DGS-inspired tricks with one flag."""
    if not opt.enable_speede_tricks:
        return
    print("[SpeeDeTricks] Enabled score-pruning + TSS + VC (ratios: 0.30/0.15).")
    opt.speede_use_tss = True
    opt.speede_use_vc = True
    # Match run_d3dgs_dnerf.sh defaults when only --enable_speede_tricks is set.
    opt.speede_densify_prune_ratio = 0.30
    opt.speede_after_densify_prune_ratio = 0.15


def _score_func_speede(
    scores, view, gaussians, opt, pipe, background, deform, time_interval, is_6dof, gflow_model=None
):
    """Compute score contribution from one camera view."""
    # Create per-Gaussian score proxies (leaf tensor) so rasterization can expose
    # gradients w.r.t. each Gaussian's contribution in this single view.
    img_scores = torch.zeros_like(scores, requires_grad=True)

    # Normalized frame timestamp for this camera.
    fid = view.fid
    # Current number of Gaussians.
    n_pts = gaussians.get_xyz.shape[0]
    # Broadcast timestamp to per-Gaussian time input shape: (N, 1).
    time_input = fid.unsqueeze(0).expand(n_pts, -1)
    # TSS: optional timestamp jitter scaled by frame interval to probe nearby times.
    tss_noise = (
        # Draw one Gaussian noise sample, broadcast to all Gaussians, then scale.
        torch.randn(1, 1, device="cuda").expand(n_pts, -1) * time_interval
        # If TSS is disabled, use exact timestamp (no perturbation).
        if opt.speede_use_tss else 0.0
    )
    
    ## TODO: Calculate scores for each Gaussian

    # Deformation is inference-only during scoring (no gradients through motion net).
    with torch.no_grad():
        # If GroupFlow is active, use grouped rigid motion instead of deform MLP.
        if gflow_model is not None:
            d_xyz, d_rotation, d_scaling = step_group_flow(gflow_model, fid, gaussians)
        # Otherwise use deform network at (optionally) jittered timestamp.
        else:
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + tss_noise)

    # Render this view while injecting `img_scores` into the rasterizer so that
    # image gradients can backpropagate to per-Gaussian score slots.
    render_pkg = render(
        view, gaussians, pipe, background,
        d_xyz, d_rotation, d_scaling, is_6dof,
        scores=img_scores)
    # RGB rendering result for this camera.
    image = render_pkg["render"]
    # Visibility mask of Gaussians that contributed in this view.
    vis = render_pkg["visibility_filter"]

    # Scalarize output and backprop to obtain d(sum(image))/d(img_scores).
    image.sum().backward()
    # Preferred path: rasterizer exposes score gradients directly.
    if img_scores.grad is not None:
        # Accumulate this view's sensitivity into global running scores.
        with torch.no_grad():
            scores += img_scores.grad
    else:
        # Fallback for rasterizers without score-gradient output.
        with torch.no_grad():
            # Visibility-count proxy: give +1 score to visible Gaussians.
            scores[vis] += 1.0

    ## TODO end of score calculation
    
    # Return visibility for optional view-count normalization in caller.
    return vis


def _prune_speede(scene, gaussians, dataset, opt, pipe, background, deform, time_interval, prune_ratio, gflow_model=None):
    """SpeeDe3DGS-style score pruning pass."""
    if gaussians.get_xyz.shape[0] <= 1024:
        return
    with torch.enable_grad():
        pbar = tqdm(total=len(scene.getTrainCameras()), desc="SpeeDe score pruning")
        scores = torch.zeros_like(gaussians.get_opacity)
        view_counts = torch.zeros_like(scores, dtype=torch.int32)
        for view in scene.getTrainCameras():
            if dataset.load2gpu_on_the_fly:
                view.load2device()
            vis = _score_func_speede(
                scores, view, gaussians, opt, pipe, background, deform,
                time_interval, dataset.is_6dof, gflow_model=gflow_model)
            with torch.no_grad():
                view_counts[vis] += 1
            if dataset.load2gpu_on_the_fly:
                view.load2device('cpu')
            if gaussians.optimizer is not None:
                gaussians.optimizer.zero_grad(set_to_none=True)
            if deform.optimizer is not None:
                deform.optimizer.zero_grad()
            pbar.update(1)
        pbar.close()

        if opt.speede_use_vc:
            norm = torch.sqrt(view_counts.float() + 1e-6)
            scores = scores / norm
            scores[view_counts == 0] = -float("inf")

        gaussians.prune_gaussians(prune_ratio, scores)


# ── Single-stage training loop ────────────────────────────────────────────────

def scene_reconstruction(dataset, opt, hyper, pipe,
                          testing_iterations, saving_iterations,
                          gaussians, scene, deform,
                          stage, tb_writer, train_iter):
    """One training stage (coarse or fine).

    Iteration counter restarts from 1 to train_iter, matching the original
    4DGaussians design where each stage has its own fresh optimizer and
    independent learning-rate schedule.

    Args:
        stage:      "coarse" (pure 3DGS) or "fine" (4DGS with deformation)
        train_iter: number of iterations for this stage
    Returns:
        (best_psnr, best_iteration) from evaluations during this stage
    """
    # ── Fresh optimizer for this stage ───────────────────────────────────────
    gaussians.training_setup(opt)

    if stage == "fine" and dataset.method == "4dgs":
        # Deform params join the same optimizer → one unified Adam
        _add_deform_to_optimizer(gaussians, deform, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(train_iter), desc=f"{stage.capitalize()} stage")

    for iteration in range(1, train_iter + 1):

        # ── Network GUI ───────────────────────────────────────────────────────
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, \
                    keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background,
                                       0.0, 0.0, 0.0)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < train_iter) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        # ── LR update BEFORE forward pass (matches original line 138) ────────
        gaussians.update_learning_rate(iteration)
        if stage == "fine" and dataset.method == "4dgs":
            _set_unified_deform_learning_rates(gaussians, deform, iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── Sample a random training view ─────────────────────────────────────
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        # ── Deformation (fine stage only) ─────────────────────────────────────
        if stage == "fine":
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling = deform.step(
                gaussians.get_xyz, time_input, gaussians=gaussians)
        else:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0

        # ── Render ────────────────────────────────────────────────────────────
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background,
            d_xyz, d_rotation, d_scaling, dataset.is_6dof,
            absolute_deform=(stage == "fine" and dataset.method == "4dgs"))
        image             = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii             = render_pkg["radii"]

        # ── Photometric loss ──────────────────────────────────────────────────
        gt_image = viewpoint_cam.original_image.cuda()
        image_for_loss = image
        if dataset.method == "4dgs" and int(getattr(opt, "stabilize_4dgs_loss", 1)) != 0:
            image_for_loss = torch.clamp(
                torch.nan_to_num(image, nan=0.5, posinf=1.0, neginf=0.0),
                0.0, 1.0,
            )

        ## TODO: Implement L1 loss and SSIM loss
        # Supervise rendered RGB with the standard 3DGS photometric objective:
        #   loss_photo = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
        # L1 captures per-pixel color fidelity; SSIM encourages structural similarity.
        Ll1 = l1_loss(image_for_loss, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image_for_loss, gt_image)
        )

        # 4DGS-only regularization (fine stage):
        # adds HexPlane priors (spatial smoothness + temporal smoothness + time-plane L1)
        # to stabilize dynamic deformation and reduce flicker/noise.
        if stage == "fine" and dataset.method == "4dgs":
            reg = deform.compute_regulation()
            if torch.isfinite(reg).all():
                loss = loss + reg
        ## TODO: End of 4DGS HexPlane TV regularisation

        # ── NaN guard: skip backward if loss invalid (original restarts; ──────
        # ── we skip to preserve the last good weights)                   ──────
        if not torch.isfinite(loss):
            print(f"\n[ITER {iteration}] NaN/Inf loss — skipping step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            iter_end.record()
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')
            continue

        loss.backward()
        clip_norm = getattr(opt, "grad_clip_norm_4dgs", 0.0)
        if dataset.method == "4dgs" and clip_norm and clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in gaussians.optimizer.param_groups for p in g["params"]],
                max_norm=float(clip_norm),
            )
        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # ── Logging ───────────────────────────────────────────────────────
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # ── Evaluation & saving ───────────────────────────────────────────
            cur_psnr = training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background),
                deform, dataset, stage)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # ── Densification ─────────────────────────────────────────────────
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                gaussians.add_densification_stats(
                    viewspace_point_tensor_densify, visibility_filter)

                # Stage-dependent thresholds (matches original 4DGaussians)
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = (
                        opt.opacity_threshold_fine_init
                        - iteration * (opt.opacity_threshold_fine_init
                                       - opt.opacity_threshold_fine_after)
                        / opt.densify_until_iter
                    )
                    densify_threshold = (
                        opt.densify_grad_threshold_fine_init
                        - iteration * (opt.densify_grad_threshold_fine_init
                                       - opt.densify_grad_threshold_after)
                        / opt.densify_until_iter
                    )

                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                        and gaussians.get_xyz.shape[0] < 360000):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold,
                                      scene.cameras_extent, size_threshold)

                if (iteration > opt.pruning_from_iter
                        and iteration % opt.pruning_interval == 0
                        and gaussians.get_xyz.shape[0] > 200000):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold,
                                    scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background
                            and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # ── Unified optimizer step ────────────────────────────────────────
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    return best_psnr, best_iteration


# ── Deformable-3DGS training (original single-stage loop) ────────────────────

def training_deformable(dataset, opt, pipe, hyper, testing_iterations, saving_iterations,
                         gaussians, scene, deform, tb_writer):
    """Single-stage training for the Deformable-3DGS method."""
    gaussians.training_setup(opt)
    deform.spatial_lr_scale = scene.cameras_extent
    deform.train_setting(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15,
                                        lr_delay_mult=0.01, max_steps=20000)
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    gflow_model = None

    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, \
                    keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background,
                                       0.0, 0.0, 0.0)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if (
            opt.gflow_flag
            and gflow_model is None
            and iteration == int(opt.gflow_iteration)
        ):
            print("[GroupFlow] Building grouped motion model ...")
            gflow_model = do_group_flow(gaussians, opt, dataset, scene, deform)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        elif gflow_model is not None:
            fid_input = fid
            if opt.gflow_noise_flag and not dataset.is_blender:
                fid_input = torch.clamp(fid + torch.randn(1, device='cuda') * time_interval * smooth_term(iteration), 0.0, 1.0)
            d_xyz, d_rotation, d_scaling = step_group_flow(gflow_model, fid_input, gaussians)
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            ast_noise = (
                0 if dataset.is_blender
                else torch.randn(1, 1, device='cuda').expand(N, -1)
                     * time_interval * smooth_term(iteration)
            )
            d_xyz, d_rotation, d_scaling = deform.step(
                gaussians.get_xyz, time_input + ast_noise)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image             = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii             = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1  = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()
        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            cur_psnr = training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background),
                deform, dataset, "fine", gflow_model=gflow_model, opt=opt)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)
                if gflow_model is not None and opt.gflow_flag:
                    gflow_model.save_weights(dataset.model_path, iteration)

            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                gaussians.add_densification_stats(
                    viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background
                            and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if (
                opt.enable_speede_tricks
                and iteration >= opt.speede_prune_from_iter
                and iteration < opt.speede_prune_until_iter
                and iteration % max(1, opt.speede_prune_interval) == 0
            ):
                prune_ratio = (
                    opt.speede_densify_prune_ratio
                    if iteration < opt.densify_until_iter
                    else opt.speede_after_densify_prune_ratio
                )
                _prune_speede(
                    scene, gaussians, dataset, opt, pipe, background, deform,
                    time_interval, prune_ratio, gflow_model=gflow_model)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if iteration >= opt.warm_up:
                    deform.optimizer.step()
                    deform.update_learning_rate(iteration)
                if gflow_model is not None and opt.gflow_flag:
                    gflow_model.optimizer.step()
                    gflow_model.optimizer.zero_grad()
                    if gflow_model.annealing_lr_flag:
                        gflow_model.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()

    print(f"Best PSNR = {best_psnr} in Iteration {best_iteration}")


# ── Top-level training dispatcher ─────────────────────────────────────────────

def training(dataset, opt, pipe, hyper, testing_iterations, saving_iterations, run_args=None):
    # Store full CLI args when available for reproducibility.
    log_args = run_args if run_args is not None else dataset
    tb_writer = prepare_output_and_logger(log_args)
    if run_args is not None:
        dataset.model_path = run_args.model_path

    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(
        method=dataset.method,
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        args=hyper,
    )

    scene = Scene(dataset, gaussians)
    _apply_speede_tricks_preset(opt)

    if dataset.method == "4dgs":
        # Schedulers for MLP + grid (used to set LR on unified optimizer each fine step).
        deform.spatial_lr_scale = gaussians.spatial_lr_scale
        deform.train_setting(opt)

        # Set HexPlane bounding box from initial point cloud
        xyz_np = gaussians.get_xyz.detach().cpu().numpy()
        deform.deform.deformation_net.set_aabb(
            xyz_np.max(axis=0).tolist(),
            xyz_np.min(axis=0).tolist())

        # ── Stage 1: Coarse (pure 3DGS, no deformation) ──────────────────────
        print(f"\n{'='*60}")
        print(f"  4DGS Stage 1: Coarse  (0 → {opt.coarse_iterations} iterations)")
        print(f"{'='*60}")

        # Evaluate once at the end of coarse stage; save nothing
        coarse_test_iters  = [opt.coarse_iterations]
        coarse_save_iters  = []

        scene_reconstruction(
            dataset, opt, hyper, pipe,
            coarse_test_iters, coarse_save_iters,
            gaussians, scene, deform,
            "coarse", tb_writer, opt.coarse_iterations)

        # ── Stage 2: Fine (4DGS with HexPlane deformation) ───────────────────
        print(f"\n{'='*60}")
        print(f"  4DGS Stage 2: Fine  (0 → {opt.iterations} iterations)")
        print(f"{'='*60}")

        best_psnr, best_iter = scene_reconstruction(
            dataset, opt, hyper, pipe,
            testing_iterations, saving_iterations,
            gaussians, scene, deform,
            "fine", tb_writer, opt.iterations)

        print(f"\nBest PSNR = {best_psnr:.4f} dB in Fine Stage Iteration {best_iter}")

    else:
        # Deformable-3DGS: single continuous training loop
        training_deformable(
            dataset, opt, pipe, hyper,
            testing_iterations, saving_iterations,
            gaussians, scene, deform, tb_writer)


# ── Utilities ─────────────────────────────────────────────────────────────────

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    args_dict = vars(args)
    # Keep cfg_args eval-compatible while making it human-readable.
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write("Namespace(\n")
        for key in sorted(args_dict.keys()):
            cfg_log_f.write(f"    {key}={pformat(args_dict[key])},\n")
        cfg_log_f.write(")\n")
    # Add JSON copy for easier downstream parsing/experiment diffing.
    with open(os.path.join(args.model_path, "cfg_args.json"), 'w') as cfg_json_f:
        json.dump(args_dict, cfg_json_f, indent=2, sort_keys=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs,
                    deform, dataset, stage, gflow_model=None, opt=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts    = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if dataset.load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

                    if gflow_model is not None and opt is not None:
                        d_xyz, d_rotation, d_scaling = step_group_flow(
                            gflow_model, fid, scene.gaussians,
                            use_precomputed_interp=True)
                    elif stage == "fine":
                        d_xyz, d_rotation, d_scaling = deform.step(
                            xyz.detach(), time_input, gaussians=scene.gaussians)
                    else:
                        d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0

                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                   d_xyz, d_rotation, d_scaling,
                                   dataset.is_6dof,
                                   absolute_deform=(stage == "fine"
                                                    and dataset.method == "4dgs"))["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts    = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if dataset.load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + f"_view_{viewpoint.image_name}/render",
                            image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + f"_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None], global_step=iteration)

                l1_test   = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
                      f"L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            n_pts = scene.gaussians.get_xyz.shape[0]
            tb_writer.add_scalar('total_points', n_pts, iteration)
            if n_pts > 0:
                op = scene.gaussians.get_opacity.detach().reshape(-1)
                op = torch.nan_to_num(op, nan=0.0, posinf=1.0, neginf=0.0)
                if op.numel() > 0 and torch.isfinite(op).all():
                    try:
                        tb_writer.add_histogram("scene/opacity_histogram", op, iteration)
                    except (ValueError, RuntimeError):
                        pass
        torch.cuda.empty_cache()

    return test_psnr


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip',   type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[3000, 7000, 14000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7000, 14000, 20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    print(f"Method: {args.method}")

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), hp.extract(args),
             args.test_iterations, args.save_iterations, run_args=args)

    print("\nTraining complete.")
