"""
Unified rendering / evaluation script.

Usage:
    python render.py -m <model_path> --method deformable [--iteration N] [--mode render]
    python render.py -m <model_path> --method 4dgs       [--iteration N] [--mode render]

Modes: render | time | view | all | pose | original
"""

import torch
from scene import Scene, GaussianModel
from scene.deform_model import DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams, get_combined_args
import imageio
import numpy as np
import time
from utils.rigid_utils import GroupFlowModel_v2, step_group_flow


def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration,
               views, gaussians, pipeline, background, deform, method="deformable",
               opt=None, gflow_model=None):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path    = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    depth_path  = os.path.join(model_path, name, f"ours_{iteration}", "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path,    exist_ok=True)
    makedirs(depth_path,  exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid        = view.fid
        xyz        = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        if gflow_model is not None and opt is not None:
            d_xyz, d_rotation, d_scaling = step_group_flow(
                gflow_model, fid, gaussians, use_precomputed_interp=True)
        else:
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, gaussians=gaussians)
        results  = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof,
                          absolute_deform=(method == "4dgs"))
        rendering = results["render"]
        depth     = results["depth"]
        depth     = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt,        os.path.join(gts_path,    f"{idx:05d}.png"))
        torchvision.utils.save_image(depth,     os.path.join(depth_path,  f"{idx:05d}.png"))

    # FPS benchmark
    for idx, view in enumerate(tqdm(views, desc="FPS benchmark")):
        fid        = view.fid
        xyz        = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        torch.cuda.synchronize()
        t_start = time.time()
        if gflow_model is not None and opt is not None:
            d_xyz, d_rotation, d_scaling = step_group_flow(
                gflow_model, fid, gaussians, use_precomputed_interp=True)
        else:
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, gaussians=gaussians)
        render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof,
               absolute_deform=(method == "4dgs"))
        torch.cuda.synchronize()
        t_list.append(time.time() - t_start)

    t   = np.array(t_list[5:])
    fps = 1.0 / t.mean() if t.size > 0 else float('nan')
    xyz = gaussians.get_xyz
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')


def interpolate_time(model_path, load2gpu_on_the_fly, is_6dof, name, iteration,
                     views, gaussians, pipeline, background, deform, method="deformable",
                     opt=None, gflow_model=None):
    render_path = os.path.join(model_path, name, f"interpolate_{iteration}", "renders")
    makedirs(render_path, exist_ok=True)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx   = torch.randint(0, len(views), (1,)).item()
    view  = views[idx]
    renderings = []
    for t in tqdm(range(frame), desc="Time interpolation"):
        fid        = torch.Tensor([t / (frame - 1)]).cuda()
        xyz        = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        if gflow_model is not None and opt is not None:
            d_xyz, d_rotation, d_scaling = step_group_flow(
                gflow_model, fid, gaussians, use_precomputed_interp=True)
        else:
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, gaussians=gaussians)
        results   = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof,
                           absolute_deform=(method == "4dgs"))
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{t:05d}.png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, hyper, opt, iteration: int,
                pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene     = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform = DeformModel(
            method=dataset.method,
            is_blender=dataset.is_blender,
            is_6dof=dataset.is_6dof,
            args=hyper,
        )
        deform.load_weights(dataset.model_path)
        # NOTE: do NOT call set_aabb after load_weights for 4DGS.
        # The AABB is saved in the checkpoint (deformation_net.grid.aabb) and
        # restored by load_state_dict. Recomputing from trained Gaussian positions
        # would give a different range and break the HexPlane coordinate normalization.

        bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_func = render_set if mode == "render" else interpolate_time
        gflow_model = None
        if opt.gflow_flag:
            try:
                gflow_model = GroupFlowModel_v2()
                gflow_model.load_weights(dataset.model_path, iteration=scene.loaded_iter)
                if hasattr(gflow_model, "precompute_interp"):
                    try:
                        gflow_model.precompute_interp(x=gaussians.get_xyz)
                    except TypeError:
                        gflow_model.precompute_interp()
            except Exception as exc:
                print(f"[GroupFlow] Failed to load weights, fallback to deform net: {exc}")
                gflow_model = None

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly,
                        dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, method=dataset.method, opt=opt, gflow_model=gflow_model)
        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly,
                        dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform, method=dataset.method, opt=opt, gflow_model=gflow_model)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp       = ModelHiddenParams(parser)
    op       = OptimizationParams(parser)
    parser.add_argument("--iteration",  default=-1,       type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test",  action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--mode",       default='render',
                        choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), hp.extract(args), op.extract(args), args.iteration,
                pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
