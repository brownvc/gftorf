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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, PA2SH, SH2PA
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from arguments import ModelParams, OptimizationParams
from scene.deform_model import *

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args : ModelParams):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc_color = torch.empty(0)
        self._features_rest_color = torch.empty(0)
        self._features_dc_phase = torch.empty(0)
        self._features_rest_phase = torch.empty(0)
        self._features_dc_amp = torch.empty(0)
        self._features_rest_amp = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_seg_color = torch.empty(0)
        self._phase_offset = torch.empty(0)
        self._dc_offset = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.scene_extent = 0
        self.setup_functions()
        
        self.dynamic = args.dynamic
        self.deform_model = DeformModel(args)

        self.use_view_dependent_phase = args.use_view_dependent_phase

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc_color,
            self._features_rest_color,
            self._features_dc_phase,
            self._features_rest_phase,
            self._features_dc_amp,
            self._features_rest_amp,
            self._scaling,
            self._rotation,
            self._opacity,
            self._features_seg_color,
            self._phase_offset,
            self._dc_offset,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.scene_extent,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc_color,
        self._features_rest_color,
        self._features_dc_phase,
        self._features_rest_phase,
        self._features_dc_amp,
        self._features_rest_amp,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._features_seg_color,
        self._phase_offset,
        self._dc_offset,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.scene_extent) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.isotropic:
            return self.scaling_activation(self._scaling.repeat(1, 3))
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_xyz_normalized(self):
        return self._xyz / self.scene_extent

    @property
    def get_features_color(self):
        features_dc_color = self._features_dc_color
        features_rest_color = self._features_rest_color
        return torch.cat((features_dc_color, features_rest_color), dim=1)
    
    @property
    def get_features_phasor(self):
        features_dc_phase = self._features_dc_phase
        features_rest_phase = self._features_rest_phase
        features_dc_amp = self._features_dc_amp
        features_rest_amp = self._features_rest_amp
        return torch.cat((torch.cat((features_dc_phase, features_dc_amp), dim=-1), torch.cat((features_rest_phase, features_rest_amp), dim=-1)), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_motion_mask(self):
        return (self.get_features_seg_color[:, 0] > 0.5).detach()

    @property
    def get_features_seg_color(self):
        return self._features_seg_color
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def query_dmlp(self, fid): # fid \in [0,1]
        t = torch.tensor(np.array([fid])).float().cuda().unsqueeze(0).expand(self.get_xyz[self.get_motion_mask].shape[0], -1)
        xyz = self.get_xyz_normalized[self.get_motion_mask].detach()
        d_xyz, d_rot, d_sh, d_sh_p = self.deform_model.deform(xyz, t)
        return d_xyz, d_rot, d_sh, d_sh_p

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cameras_extent, scene_extent : float, args):
        self.isotropic = args.isotropic_gaussians
        self.cameras_extent = cameras_extent
        self.scene_extent = scene_extent
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features_color = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_color[:, :3, 0 ] = fused_color
        features_color[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        if args.init_static_first:
            dist2_static = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[:fused_point_cloud.shape[0]//2, :])).float().cuda()), 0.0000001)
            dist2_dynamic = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[fused_point_cloud.shape[0]//2:, :])).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(torch.cat([dist2_static, dist2_dynamic], dim=0)))[...,None]
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(args.initial_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc_color = nn.Parameter(features_color[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_color = nn.Parameter(features_color[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        if self.isotropic:
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(False))
        else:
            self._scaling = nn.Parameter(scales.repeat(1, 3).requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if pcd.phases is not None:
            phase = PA2SH(torch.tensor(np.asarray(pcd.phases)).float().cuda())
            features_phase = torch.zeros((phase.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features_phase[:, :1, 0 ] = phase
            features_phase[:, 1:, 1:] = 1.0
            self._features_dc_phase = nn.Parameter(features_phase[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest_phase = nn.Parameter(features_phase[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._phase_offset = nn.Parameter(torch.zeros((1)).float().cuda().requires_grad_(True))
            self._dc_offset = nn.Parameter(torch.zeros((1)).float().cuda().requires_grad_(True))
        if pcd.amplitudes is not None:
            amp = PA2SH(torch.tensor(np.asarray(pcd.amplitudes)).float().cuda())
            features_amp = torch.zeros((amp.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features_amp[:, :1, 0 ] = amp
            features_amp[:, 1:, 1:] = 0.0
            self._features_dc_amp = nn.Parameter(features_amp[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest_amp = nn.Parameter(features_amp[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        if pcd.seg_colors is not None:
            features_seg_color = torch.tensor(np.asarray(pcd.seg_colors)).float().cuda()
            self._features_seg_color = nn.Parameter(features_seg_color.contiguous().requires_grad_(False))

    def training_setup(self, training_args):
        if self.isotropic:
            training_args.rotation_lr = 0.0
        self.deform_model.training_setup(training_args, self.cameras_extent)

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.scene_extent, "name": "xyz"}
        ]

        l += [
            {'params': [self._features_dc_color], 'lr': training_args.feature_lr, "name": "f_dc_color"},
            {'params': [self._features_rest_color], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_color"}
            ]
        l += [
            {'params': [self._features_dc_phase], 'lr': training_args.feature_phase_lr_init * self.scene_extent, "name": "phase_f_dc"},
            {'params': [self._features_rest_phase], 'lr': training_args.feature_phase_lr_init * self.scene_extent / 20.0, "name": "phase_f_rest"},
            {'params': [self._features_dc_amp], 'lr': training_args.feature_amp_lr_init * self.scene_extent**2, "name": "amp_f_dc"},
            {'params': [self._features_rest_amp], 'lr': training_args.feature_amp_lr_init * self.scene_extent**2 / 20.0, "name": "amp_f_rest"},
        ]

        l += [
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._features_seg_color], 'lr': 0.0, "name": "f_seg_color"},
        ]

        l += [
            {'params': [self._phase_offset], 'lr': 0.0, "name": "phase_offset"},
            {'params': [self._dc_offset], 'lr': 0.0, "name": "dc_offset"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.scene_extent,
            lr_final=training_args.position_lr_final*self.scene_extent,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

        self.phase_scheduler_args = get_expon_lr_func(
            lr_init=training_args.feature_phase_lr_init*self.scene_extent,
            lr_final=training_args.feature_phase_lr_final*self.scene_extent,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        
        self.amp_scheduler_args = get_expon_lr_func(
            lr_init=training_args.feature_amp_lr_init * self.scene_extent**2,
            lr_final=training_args.feature_amp_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, args : OptimizationParams):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            if param_group["name"] == "xyz" or param_group["name"].startswith("delta_xyz"):
                lr = self.xyz_scheduler_args(iteration)
            if param_group["name"] in ["phase_f_dc"]:
                lr = self.phase_scheduler_args(iteration)
            if param_group["name"] in ["phase_f_rest"]:
                lr = self.phase_scheduler_args(iteration)
            if param_group["name"] in ["amp_f_dc"]:
                lr = self.amp_scheduler_args(iteration)
            if param_group["name"] in ["amp_f_rest"]:
                lr = self.amp_scheduler_args(iteration)
            if iteration > args.optimize_offset_start:
                if param_group["name"] in ["phase_offset"]:
                    lr = args.phase_offset_lr
                if param_group["name"] in ["dc_offset"]:
                    lr = args.dc_offset_lr
            param_group['lr'] = lr

    def construct_list_of_attributes(self, save_all=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc_color.shape[1]*self._features_dc_color.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest_color.shape[1]*self._features_rest_color.shape[2]):
            l.append('f_rest_{}'.format(i)) 
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if save_all:
            for i in range(self._features_dc_phase.shape[1]*self._features_dc_phase.shape[2]):
                l.append('phase_f_dc_{}'.format(i))
            for i in range(self._features_rest_phase.shape[1]*self._features_rest_phase.shape[2]):
                l.append('phase_f_rest_{}'.format(i))
            for i in range(self._features_dc_amp.shape[1]*self._features_dc_amp.shape[2]):
                l.append('amp_f_dc_{}'.format(i))
            for i in range(self._features_rest_amp.shape[1]*self._features_rest_amp.shape[2]):
                l.append('amp_f_rest_{}'.format(i))     
            for i in range(self._features_seg_color.shape[1]):
                l.append('f_seg_color_{}'.format(i))   
        return l

    def save_ply(self, path, sibr_only=True):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc_color = self._features_dc_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_color = self._features_rest_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        if sibr_only:
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc_color, f_rest_color, opacities, scale, rotation), axis=1)
        else:
            phase_f_dc = self._features_dc_phase.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            phase_f_rest = self._features_rest_phase.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            amp_f_dc = self._features_dc_amp.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            amp_f_rest = self._features_rest_amp.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_seg_color = self._features_seg_color.detach().contiguous().cpu().numpy()
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(save_all=True)]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc_color, f_rest_color, opacities, scale, rotation, phase_f_dc, phase_f_rest, amp_f_dc, amp_f_rest, f_seg_color), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, apply_mask=None):
        if apply_mask is None:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        else:
            opacities_new = self._opacity.clone()
            opacities_new[apply_mask] = inverse_sigmoid(torch.min(self.get_opacity[apply_mask], torch.ones_like(self.get_opacity[apply_mask])*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc_color = np.zeros((xyz.shape[0], 3, 1))
        features_dc_color[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc_color[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc_color[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names_color = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names_color = sorted(extra_f_names_color, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_color)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_color = np.zeros((xyz.shape[0], len(extra_f_names_color)))
        for idx, attr_name in enumerate(extra_f_names_color):
            features_extra_color[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_color = features_extra_color.reshape((features_extra_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        features_dc_phase = np.zeros((xyz.shape[0], 1, 1))
        features_dc_phase[:, 0, 0] = np.asarray(plydata.elements[0]["phase_f_dc_0"])
        
        extra_f_names_phase = [p.name for p in plydata.elements[0].properties if p.name.startswith("phase_f_rest_")]
        extra_f_names_phase = sorted(extra_f_names_phase, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_phase)==(self.max_sh_degree + 1) ** 2 - 1
        features_extra_phase = np.zeros((xyz.shape[0], len(extra_f_names_phase)))
        for idx, attr_name in enumerate(extra_f_names_phase):
            features_extra_phase[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra_phase = features_extra_phase.reshape((features_extra_phase.shape[0], 1, (self.max_sh_degree + 1) ** 2 - 1))
        
        features_dc_amp = np.zeros((xyz.shape[0], 1, 1))
        features_dc_amp[:, 0, 0] = np.asarray(plydata.elements[0]["amp_f_dc_0"])

        extra_f_names_amp = [p.name for p in plydata.elements[0].properties if p.name.startswith("amp_f_rest_")]
        extra_f_names_amp = sorted(extra_f_names_amp, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_amp)==(self.max_sh_degree + 1) ** 2 - 1
        features_extra_amp = np.zeros((xyz.shape[0], len(extra_f_names_amp)))
        for idx, attr_name in enumerate(extra_f_names_amp):
            features_extra_amp[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra_amp = features_extra_amp.reshape((features_extra_amp.shape[0], 1, (self.max_sh_degree + 1) ** 2 - 1))
        
        f_seg_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_seg_color_")]
        f_seg_color_names = sorted(f_seg_color_names, key = lambda x: int(x.split('_')[-1]))
        f_seg_colors = np.zeros((xyz.shape[0], len(f_seg_color_names)))
        for idx, attr_name in enumerate(f_seg_color_names):
            f_seg_colors[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self._features_dc_color = nn.Parameter(torch.tensor(features_dc_color, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_color = nn.Parameter(torch.tensor(features_extra_color, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_phase = nn.Parameter(torch.tensor(features_dc_phase, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_phase = nn.Parameter(torch.tensor(features_extra_phase, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_amp = nn.Parameter(torch.tensor(features_dc_amp, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_amp = nn.Parameter(torch.tensor(features_extra_amp, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_seg_color = nn.Parameter(torch.tensor(f_seg_colors, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ['phase_offset', 'dc_offset']:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ['phase_offset', 'dc_offset']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_color = optimizable_tensors["f_dc_color"]
        self._features_rest_color = optimizable_tensors["f_rest_color"]
        self._features_dc_phase = optimizable_tensors["phase_f_dc"]
        self._features_rest_phase = optimizable_tensors["phase_f_rest"]
        self._features_dc_amp = optimizable_tensors["amp_f_dc"]
        self._features_rest_amp = optimizable_tensors["amp_f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_seg_color = optimizable_tensors["f_seg_color"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ['phase_offset', 'dc_offset']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc_color, new_features_rest_color, new_features_dc_phase, new_features_rest_phase, new_features_dc_amp, new_features_rest_amp, new_opacities, new_scaling, new_rotation, new_features_seg_color):
        d = {"xyz": new_xyz}
        d["f_dc_color"] = new_features_dc_color
        d["f_rest_color"] = new_features_rest_color
        d["phase_f_dc"] = new_features_dc_phase
        d["phase_f_rest"] = new_features_rest_phase
        d["amp_f_dc"] = new_features_dc_amp
        d["amp_f_rest"] = new_features_rest_amp
        d["opacity"] = new_opacities
        d["scaling"] = new_scaling
        d["rotation"] = new_rotation
        d["f_seg_color"] = new_features_seg_color

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_color = optimizable_tensors["f_dc_color"]
        self._features_rest_color = optimizable_tensors["f_rest_color"]
        self._features_dc_phase = optimizable_tensors["phase_f_dc"]
        self._features_rest_phase = optimizable_tensors["phase_f_rest"]
        self._features_dc_amp = optimizable_tensors["amp_f_dc"]
        self._features_rest_amp = optimizable_tensors["amp_f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_seg_color = optimizable_tensors["f_seg_color"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if self.isotropic:
            new_scaling = self.scaling_inverse_activation(self.scaling_activation(self._scaling[selected_pts_mask]).repeat(N,1) / (0.8*N))
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        new_features_dc_color = self._features_dc_color[selected_pts_mask].repeat(N,1,1)
        new_features_rest_color = self._features_rest_color[selected_pts_mask].repeat(N,1,1)
        new_features_dc_phase = self._features_dc_phase[selected_pts_mask].repeat(N,1,1)
        new_features_rest_phase = self._features_rest_phase[selected_pts_mask].repeat(N,1,1)
        new_features_dc_amp = self._features_dc_amp[selected_pts_mask].repeat(N,1,1)
        new_features_rest_amp = self._features_rest_amp[selected_pts_mask].repeat(N,1,1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_features_seg_color = self._features_seg_color[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc_color, new_features_rest_color, new_features_dc_phase, new_features_rest_phase, new_features_dc_amp, new_features_rest_amp, new_opacity, new_scaling, new_rotation, new_features_seg_color)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc_color = self._features_dc_color[selected_pts_mask]
        new_features_rest_color = self._features_rest_color[selected_pts_mask]
        new_features_dc_phase = self._features_dc_phase[selected_pts_mask]
        new_features_rest_phase = self._features_rest_phase[selected_pts_mask]
        new_features_dc_amp = self._features_dc_amp[selected_pts_mask]
        new_features_rest_amp = self._features_rest_amp[selected_pts_mask]

        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_features_seg_color = self._features_seg_color[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc_color, new_features_rest_color, new_features_dc_phase, new_features_rest_phase, new_features_dc_amp, new_features_rest_amp, new_opacities, new_scaling, new_rotation, new_features_seg_color)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size=20):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0 

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # prune_mask = torch.logical_or(prune_mask, (SH2PA(self.get_features_phasor[:, 0, 1]) <= 0.0).squeeze())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.05 * extent
            small_points_ws = self.get_scaling.max(dim=1).values < 0.001 * extent
            prune_mask = torch.logical_or(torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws), small_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def prune(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # prune_mask = torch.logical_or(prune_mask, (SH2PA(self.get_features_phasor[:, 0, 1]) <= 0.0).squeeze())
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels, apply_mask=None):
        if apply_mask is None:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * pixels[update_filter]
            self.denom[update_filter] += pixels[update_filter]
        else:
            self.xyz_gradient_accum[torch.logical_and(apply_mask, update_filter)] += torch.norm(viewspace_point_tensor.grad[torch.logical_and(apply_mask, update_filter),:2], dim=-1, keepdim=True) * pixels[update_filter]
            self.denom[torch.logical_and(apply_mask, update_filter)] += pixels[update_filter]

    def save_phase_offset(self, path):
        np.save(path, self._phase_offset.detach().contiguous().cpu().numpy())

    def load_phase_offset(self, path):
        self._phase_offset = nn.Parameter(torch.tensor(np.load(path), dtype=torch.float, device="cuda").requires_grad_(True)) 

    def save_dc_offset(self, path):
        np.save(path, self._dc_offset.detach().contiguous().cpu().numpy())

    def load_dc_offset(self, path):
        self._dc_offset = nn.Parameter(torch.tensor(np.load(path), dtype=torch.float, device="cuda").requires_grad_(True)) 

    @property
    def get_phase_offset(self):
        return self._phase_offset

    @property
    def get_dc_offset(self):
        return self._dc_offset
    