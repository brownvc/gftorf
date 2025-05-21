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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from typing import Optional

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    near_n: Optional[float] = 0.01
    far_n: Optional[float] = 100.0
    depth_range: Optional[float] = 100.0
    use_view_dependent_phase: Optional[bool] = False
    optimize_phase_offset: Optional[bool] = False
    optimize_dc_offset: Optional[bool] = False

def rasterize_gaussians(
    means3D,
    means2D,
    sh, sh_p,
    colors_precomp, phasors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    phase_offset,
    dc_offset,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh, sh_p,
        colors_precomp, phasors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        phase_offset,
        dc_offset,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh, sh_p, # Could be empty
        colors_precomp, phasors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        phase_offset,
        dc_offset,
        raster_settings : GaussianRasterizationSettings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp, phasors_precomp, # Could be EMPTY
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh, sh_p,                   # Could be EMPTY
            raster_settings.sh_degree,  # active sh degree
            raster_settings.campos,     # camera center
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.near_n,
            raster_settings.far_n,
            raster_settings.depth_range,
            raster_settings.use_view_dependent_phase,
            phase_offset.detach().cpu().numpy().item() if raster_settings.optimize_phase_offset else phase_offset,
            dc_offset.detach().cpu().numpy().item() if raster_settings.optimize_dc_offset else dc_offset,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, phasor, depth, normal, acc, entropy, depth_distortion, amp_distortion, pixels, distribution, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, phasor, depth, normal, acc, entropy, depth_distortion, amp_distortion, pixels, distribution, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, phasors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, sh_p, geomBuffer, binningBuffer, imgBuffer)
        ctx.phase_offset = phase_offset
        ctx.dc_offset = dc_offset
        return color, phasor, depth, normal, acc, entropy, depth_distortion, amp_distortion, pixels, distribution, radii
    @staticmethod
    def backward(ctx, 
                 grad_out_color, grad_out_phasor, 
                 grad_out_depth, grad_out_normal, grad_out_acc, 
                 grad_entropy, grad_depth_distortion, grad_amp_distortion, grad_pixels, grad_distribution,
                 _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, phasors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, sh_p, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors
        phase_offset = ctx.phase_offset
        dc_offset = ctx.dc_offset

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D, 
            radii, 
            colors_precomp, phasors_precomp,
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            cov3Ds_precomp, 
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            grad_out_color, grad_out_phasor, 
            grad_out_depth, grad_out_normal, grad_out_acc, 
            grad_entropy, grad_depth_distortion, grad_amp_distortion,
            sh, sh_p,
            raster_settings.sh_degree, 
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            raster_settings.near_n,
            raster_settings.far_n,
            raster_settings.depth_range,
            raster_settings.use_view_dependent_phase,
            phase_offset,
            dc_offset,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_phasors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_sh_p, grad_scales, grad_rotations, grad_phase_offset, grad_dc_offset = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_phasors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_sh_p, grad_scales, grad_rotations, grad_phase_offset, grad_dc_offset = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh, grad_sh_p,
            grad_colors_precomp, grad_phasors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_phase_offset if raster_settings.optimize_phase_offset else None,
            grad_dc_offset if raster_settings.optimize_dc_offset else None,
            None,
        )

        return grads

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.near_n,
                raster_settings.far_n)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, shs_p = None, colors_precomp = None, phasors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, phase_offset = 0.0, dc_offset = 0.0):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        # if (shs_p is None and phasors_precomp is None) or (shs_p is not None and phasors_precomp is not None):
        #     raise Exception('Please provide excatly one of either SHs_p or precomputed phasors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if shs_p is None:
            shs_p = torch.Tensor([])
        if phasors_precomp is None:
            phasors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs, shs_p,
            colors_precomp, phasors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            phase_offset,
            dc_offset,
            raster_settings, 
        )

