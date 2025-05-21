#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
import json

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
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
                elif t == list:
                    parser.add_argument('--' + key, nargs=len(value), default=value, type=type(value[0]))
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
        self.bg_color = [0.0 for _ in range(7)]
        self.random_bg_color = False
        self.data_device = "cuda"
        self.eval = False

        # Dynamic model
        self.dynamic = False      
        self.shuffle_frames = False

        self.D = 8
        self.W = 256
        self.xyz_multires = 10
        self.t_multires = 10
        self.use_timenet = False

        # ToRF dataset
        self.dataset_type = "real"
        self.total_num_views = 30
        self.train_views = ""
        self.total_num_spiral_views = 60

        self.tof_image_width = 320
        self.tof_image_height = 240
        self.tof_scale_factor = 1.0

        self.color_image_width = 320
        self.color_image_height = 240
        self.color_scale_factor = 1.0
        
        self.min_depth_fac = 0.05
        self.max_depth_fac = 0.55
        self.depth_range = 10.0 # c/f, twice the unambiguous range of the ToF sensor
        self.phase_offset = -99.0

        self.dc_offset = 0.0
        self.tof_permutation = ""

        self.use_view_dependent_phase = False

        self.init_method = "random"
        self.num_points = 100_000 
        self.phase_resolution_stride = 2
        self.initial_opacity = 0.1
        self.initial_amplitude = 0.1

        self.quad_scale = -1.0

        self.init_static_dynamic_separation = False
        self.init_static_first = False

        self.isotropic_gaussians = False
        
        self.xavier_init_dxyz = False
        
        self.start_id = 0

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

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.acc_loss_iter_start = 0
        self.dd_loss_iter_start = 0
        self.dd_loss_iter_end = 0
        self.tof_iters = 2000
        self.warm_up = 2000
        self.flow_loss_iter_start = 2000

        # Training images
        self.lambda_color = 0.0
        self.lambda_tof = 1.0
        self.num_phasor_channels = 2
        self.lambda_depth = 0.0

        # Losses
        self.lambda_acc = 0.0
        self.lambda_dd = 0.0
        self.use_wl1c = False
        self.use_wl1p = False
        self.wl1p_e = 0.1
        self.lambda_flow = 0.01

        self.use_opacity_entropy_loss = False
        self.oe_loss_iter_start = 2000
        self.oe_loss_iter_end = 20000
        self.lambda_oe = 0.01

        self.use_scale_loss = False
        self.scale_loss_iter_start = 0
        self.scale_loss_iter_end = 20000
        self.lambda_scale = 0.1

        # Learning rates
        self.deform_lr_init = 0.0008
        self.deform_lr_final = 0.0000016
        
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016

        self.feature_phase_lr_init = 0.0
        self.feature_phase_lr_final = 0.0

        self.feature_amp_lr_init = 0.00016
        self.feature_amp_lr_final = 0.00016

        self.feature_seg_lr = 0.0
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.min_opacity = 0.01

        self.optimize_offset_start = 4000
        self.optimize_phase_offset = False
        self.phase_offset_lr = 0.000001
        self.optimize_dc_offset = False
        self.dc_offset_lr = 0.000001

        self.use_quad = False

        self.optimize_sync_iters = -1
        
        self.lambda_mlp_reg = 0.0

        super().__init__(parser, "Optimization Parameters")
    
    def extract(self, args):
        g = super().extract(args)
        return g

def get_combined_args(parser : ArgumentParser):
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
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def save_args(args, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    all_args_dict = vars(args)
    with open(os.path.join(folder, file_name), 'w') as json_file:
        json.dump(all_args_dict, json_file, indent=4)