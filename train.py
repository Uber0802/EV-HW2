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
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

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
        Ll1  = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 4DGS HexPlane TV regularisation (fine stage only)
        if stage == "fine" and dataset.method == "4dgs":
            loss = loss + deform.compute_regulation()

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
                deform, dataset, "fine")
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

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

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if iteration >= opt.warm_up:
                    deform.optimizer.step()
                    deform.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()

    print(f"Best PSNR = {best_psnr} in Iteration {best_iteration}")


# ── Top-level training dispatcher ─────────────────────────────────────────────

def training(dataset, opt, pipe, hyper, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(
        method=dataset.method,
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        args=hyper,
    )

    scene = Scene(dataset, gaussians)

    if dataset.method == "4dgs":
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
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs,
                    deform, dataset, stage):
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

                    if stage == "fine":
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
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
             args.test_iterations, args.save_iterations)

    print("\nTraining complete.")
