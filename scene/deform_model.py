"""
Unified deformation model that wraps both methods under the same interface.

Both methods expose:
    deform.step(xyz, time_emb, gaussians=None) -> (d_xyz, d_rotation, d_scaling)
    deform.train_setting(opt)
    deform.save_weights(model_path, iteration)
    deform.load_weights(model_path, iteration)
    deform.update_learning_rate(iteration)

Output convention (same as Deformable-3DGS renderer):
    d_xyz      : (N, 3)  position delta in world space
    d_rotation : (N, 4)  quaternion delta added to pc.get_rotation  (normalized)
    d_scaling  : (N, 3)  scale delta added to pc.get_scaling        (exp-activated)

For Deformable-3DGS this is the native output of DeformNetwork.
For 4DGaussians the network returns absolute (pre-activation) values;
    the adapter converts them to deltas so the renderer stays identical.
"""

import os
import torch
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, method: str, is_blender: bool = False, is_6dof: bool = False, args=None):
        """
        Args:
            method:      "deformable" or "4dgs"
            is_blender:  Deformable-3DGS flag – use blender-optimised time encoding
            is_6dof:     Deformable-3DGS flag – use SE(3) screw-axis deformation
            args:        ModelHiddenParams (required for "4dgs")
        """
        self.method = method
        self.spatial_lr_scale = 5

        if method == "deformable":
            from utils.time_utils import DeformNetwork
            self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()

        elif method == "4dgs":
            if args is None:
                raise ValueError("ModelHiddenParams (args) is required for method='4dgs'")
            from scene.deformation import deform_network
            self.deform = deform_network(args).cuda()
            # Store 4DGS-specific args for regulation loss (used by train.py)
            self.hyper = args

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'deformable' or '4dgs'.")

        self.optimizer = None

    # ── Core interface ────────────────────────────────────────────────────────

    def step(self, xyz: torch.Tensor, time_emb: torch.Tensor, gaussians=None):
        """Compute deformation deltas for all Gaussians at a given time.

        Args:
            xyz       : (N, 3)  canonical Gaussian positions (detached)
            time_emb  : (N, 1)  normalised time in [0, 1]
            gaussians : GaussianModel – required for method='4dgs'

        Returns:
            d_xyz      : (N, 3)
            d_rotation : (N, 4)
            d_scaling  : (N, 3)
        """
        if self.method == "deformable":
            # DeformNetwork returns (d_xyz, d_rotation, d_scaling) natively.
            return self.deform(xyz, time_emb)

        elif self.method == "4dgs":
            assert gaussians is not None, \
                "gaussians must be passed to DeformModel.step() when using method='4dgs'"

            raw_scales    = gaussians._scaling   # pre-activation (no detach: gradient flows through deformation)
            raw_rots      = gaussians._rotation  # pre-normalisation (no detach)
            opacity       = gaussians._opacity.detach()
            shs           = gaussians.get_features.detach()

            # Forward through the HexPlane deformation network.
            # Returns absolute (pre-activation) values for scales and rotations.
            new_xyz, new_scales_raw, new_rots_raw, _new_opacity, _new_shs = \
                self.deform(xyz, raw_scales, raw_rots, opacity, shs, time_emb)

            # Return ABSOLUTE deformed values (not deltas).
            # The renderer is called with absolute_deform=True for 4DGS, so it uses
            # these directly without adding the canonical Gaussian attributes.
            # This matches the original 4DGaussians: gradient flows only through the
            # deformation network, not via any canonical + delta cancellation path.
            abs_xyz      = new_xyz
            abs_scaling  = gaussians.scaling_activation(new_scales_raw)
            abs_rotation = gaussians.rotation_activation(new_rots_raw)

            return abs_xyz, abs_rotation, abs_scaling

    # ── Optimiser setup ───────────────────────────────────────────────────────

    def train_setting(self, training_args):
        if self.method == "deformable":
            l = [{'params': list(self.deform.parameters()),
                  'lr': training_args.position_lr_init * self.spatial_lr_scale,
                  'name': 'deform'}]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.deform_scheduler_args = get_expon_lr_func(
                lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                lr_final=training_args.position_lr_final,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.deform_lr_max_steps)

        elif self.method == "4dgs":
            # Separate LR for MLP and HexPlane grid. Schedule length matches hustvl/4DGaussians
            # gaussian_model.training_setup: deformation + grid use position_lr_max_steps (not a
            # separate deform max_steps).
            l = [
                {'params': list(self.deform.get_mlp_parameters()),
                 'lr': training_args.deformation_lr_init * self.spatial_lr_scale,
                 'name': 'deform_mlp'},
                {'params': list(self.deform.get_grid_parameters()),
                 'lr': training_args.grid_lr_init * self.spatial_lr_scale,
                 'name': 'deform_grid'},
            ]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            _lr_steps = training_args.position_lr_max_steps
            self.mlp_scheduler_args = get_expon_lr_func(
                lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
                lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
                lr_delay_mult=training_args.deformation_lr_delay_mult,
                max_steps=_lr_steps)
            self.grid_scheduler_args = get_expon_lr_func(
                lr_init=training_args.grid_lr_init * self.spatial_lr_scale,
                lr_final=training_args.grid_lr_final * self.spatial_lr_scale,
                lr_delay_mult=training_args.deformation_lr_delay_mult,
                max_steps=_lr_steps)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_weights(self, model_path: str, iteration: int):
        out_dir = os.path.join(model_path, "deform", f"iteration_{iteration}")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_dir, "deform.pth"))

    def load_weights(self, model_path: str, iteration: int = -1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform",
                                    f"iteration_{loaded_iter}", "deform.pth")
        self.deform.load_state_dict(torch.load(weights_path))

    # ── Learning-rate scheduling ──────────────────────────────────────────────

    def update_learning_rate(self, iteration: int):
        if self.method == "deformable":
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "deform":
                    lr = self.deform_scheduler_args(iteration)
                    param_group["lr"] = lr
                    return lr

        elif self.method == "4dgs":
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "deform_mlp":
                    param_group["lr"] = self.mlp_scheduler_args(iteration)
                elif param_group["name"] == "deform_grid":
                    param_group["lr"] = self.grid_scheduler_args(iteration)

    # ── 4DGaussians regulation loss ───────────────────────────────────────────

    def compute_regulation(self) -> torch.Tensor:
        """HexPlane regularisation matching original 4DGaussians gaussian_model.py.

        _plane_regulation : compute_plane_smoothness on spatial planes  [0,1,3], weight=plane_tv_weight
        _time_regulation  : compute_plane_smoothness on temporal planes  [2,4,5], weight=time_smoothness_weight
        _l1_regulation    : abs(1 - grid).mean()  on temporal planes    [2,4,5], weight=l1_time_planes
        """
        if self.method != "4dgs":
            return torch.tensor(0.0, device="cuda")
        from scene.regulation import compute_plane_smoothness
        multi_res_grids = self.deform.deformation_net.grid.grids
        plane_smooth = torch.tensor(0.0, device="cuda")   # _plane_regulation
        time_smooth  = torch.tensor(0.0, device="cuda")   # _time_regulation
        l1_planes    = torch.tensor(0.0, device="cuda")   # _l1_regulation
        # itertools.combinations(range(4), 2) order:
        #   0:(0,1)=xy  1:(0,2)=xz  2:(0,3)=xt  3:(1,2)=yz  4:(1,3)=yt  5:(2,3)=zt

        ## TODO: Implement plane smoothness and l1 regulation
        # Accumulate regularization across all multi-resolution HexPlane levels.
        # Plane index mapping (from combinations over x,y,z,t):
        #   0:xy, 1:xz, 2:xt, 3:yz, 4:yt, 5:zt
        # Spatial smoothness is applied to pure spatial planes.
        # Temporal penalties are applied to planes that include t.
        for grids in multi_res_grids:
            if len(grids) == 3:
                spatial_ids = list(range(3))
                time_ids = []
            else:
                spatial_ids = [0, 1, 3]   # pure spatial planes (xy, xz, yz)
                time_ids    = [2, 4, 5]   # spatiotemporal planes (xt, yt, zt)
            for gid in spatial_ids:
                plane_smooth += compute_plane_smoothness(grids[gid])   # _plane_regulation
            for gid in time_ids:
                time_smooth += compute_plane_smoothness(grids[gid])    # _time_regulation
                l1_planes   += torch.abs(1 - grids[gid]).mean()        # _l1_regulation
        return (self.hyper.plane_tv_weight       * plane_smooth
                + self.hyper.time_smoothness_weight * time_smooth
                + self.hyper.l1_time_planes         * l1_planes)
        ## TODO: End of plane smoothness and l1 regulation
