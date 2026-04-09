from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        # Official repo defaults eval=True; argparse bool here defaults False — use --eval for D-NeRF parity
        self.eval = False
        self.load2gpu_on_the_fly = False
        # Deformable-3DGS flags
        self.is_blender = False
        self.is_6dof = False
        # Method selector: "deformable" or "4dgs"
        self.method = "deformable"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


# ── 4DGaussians architecture hyperparameters ─────────────────────────────────
class ModelHiddenParams(ParamGroup):
    """Architecture hyperparameters for the 4DGaussians HexPlane deformation network.
    Only used when --method 4dgs is selected."""
    def __init__(self, parser):
        self.net_width = 64          # Width of deformation MLP
        self.timebase_pe = 4         # Time positional encoding frequencies
        self.defor_depth = 0         # Depth of deformation MLP (0 = single linear in feature_out)
        self.posebase_pe = 10        # Spatial positional encoding frequencies
        self.scale_rotation_pe = 2   # PE frequencies for scale/rotation inputs
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6            # Scene bounding box half-size
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        # HexPlane grid configuration
        self.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 75],  # half of ~150 dynamic frames in bouncingballs
        }
        self.multires = [1, 2]  # Multi-resolution scales for HexPlane (paper uses [1,2] for dnerf)
        self.no_dx = False            # Disable position deformation
        self.no_grid = False          # Disable HexPlane (use raw xyz+t MLP instead)
        self.no_ds = False            # Disable scale deformation
        self.no_dr = False            # Disable rotation deformation
        self.no_do = True             # Disable opacity deformation
        self.no_dshs = True           # Disable SH color deformation
        self.empty_voxel = False
        self.grid_pe = 0
        self.static_mlp = False
        self.apply_rotation = False
        super().__init__(parser, "ModelHiddenParams")


# ── Optimization hyperparameters (superset of both methods) ──────────────────
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # ── Iterations ────────────────────────────────────────────────────────
        # For 4DGS: coarse_iterations = Stage 1 length; iterations = Stage 2 length
        # For deformable: only iterations is used (warm_up splits coarse/fine)
        self.iterations = 30_000           # Fine stage iterations (matches original 4DGaussians)
        self.coarse_iterations = 3_000     # Coarse stage iterations (4DGS only)
        self.warm_up = 3_000               # Deformable method: warm-up before deformation

        # ── Position LR ───────────────────────────────────────────────────────
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000   # Matches original 4DGaussians default

        # ── Deformation MLP LR (deformable method) ────────────────────────────
        self.deform_lr_max_steps = 40_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01

        # ── HexPlane grid LR (4DGS only) ──────────────────────────────────────
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        # ── Feature / opacity / scale / rotation LR ───────────────────────────
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005   # Matches original 4DGaussians default
        self.rotation_lr = 0.001

        # ── Loss weights ──────────────────────────────────────────────────────
        self.percent_dense = 0.01
        self.lambda_dssim = 0.0
        # 4DGS only (EV-HW2 extras; upstream has no nan_to_num / grad clip). Use 0 for parity.
        self.stabilize_4dgs_loss = 0
        self.grad_clip_norm_4dgs = 0.0

        # SpeeDe3DGS-inspired acceleration tricks (deformable mode)
        # 1) Temporal sensitivity sampling (TSS) in prune scoring
        # 2) View-count normalization (VC) for pruning scores
        # 3) Score-based temporal pruning schedule
        self.enable_speede_tricks = False
        self.speede_use_tss = False
        self.speede_use_vc = False
        self.speede_prune_from_iter = 6000
        self.speede_prune_until_iter = 30000
        self.speede_prune_interval = 3000
        self.speede_densify_prune_ratio = 0.60
        self.speede_after_densify_prune_ratio = 0.30

        # SpeeDe3DGS GroupFlow options (deformable mode)
        self.gflow_flag = False
        self.gflow_iteration = 15000
        self.gflow_num = 2048
        self.gflow_opt = 2
        self.gflow_xyz_lr = 1e-5
        self.gflow_rotation_lr = 1e-7
        self.gflow_translation_lr = 1e-7
        self.gflow_radius_lr = 1e-7
        self.gflow_local_rot = False
        self.gflow_local_rot_for_train = False
        self.LBS_flag = False
        self.gflow_annealing_lr_flag = False
        self.gflow_noise_flag = True
        self.gflow_tnum_max = 200

        # ── Densification schedule ────────────────────────────────────────────
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000

        # ── Grad thresholds (stage-specific, matches original 4DGaussians) ────
        self.densify_grad_threshold = 0.0002          # deformable method
        self.densify_grad_threshold_coarse = 0.0002   # 4DGS coarse stage
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002

        # ── Opacity thresholds (stage-specific) ───────────────────────────────
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005

        # ── Pruning (separate from densification, 4DGS) ───────────────────────
        self.pruning_from_iter = 500
        self.pruning_interval = 8000   # dnerf config: infrequent pruning

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
