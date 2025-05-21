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
import math
from diff_gaussian_rasterization_w_tof import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import ToFCamera
from utils.sh_utils import eval_sh

def render(viewpoint_camera : ToFCamera, pc : GaussianModel, d_xyz, d_rot, d_sh, d_sh_p, pipe, opt, bg_color : torch.Tensor, scaling_modifier = 1.0, render_regions=["static", "dynamic"]):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings_color_view = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center, 
        prefiltered=False,
        debug=False,
        near_n=viewpoint_camera.znear,
        far_n=viewpoint_camera.zfar,
    )
    rasterizer_color = GaussianRasterizer(raster_settings=raster_settings_color_view)

    if viewpoint_camera.FoVx_tof is not None:
        tanfovx_tof = math.tan(viewpoint_camera.FoVx_tof * 0.5)
        tanfovy_tof = math.tan(viewpoint_camera.FoVy_tof * 0.5)
        raster_settings_tof_view = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.tof_image_height),
            image_width=int(viewpoint_camera.tof_image_width),
            tanfovx=tanfovx_tof,
            tanfovy=tanfovy_tof,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform_tof,
            projmatrix=viewpoint_camera.full_proj_transform_tof,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center_tof, 
            prefiltered=False,
            debug=False,
            near_n=viewpoint_camera.znear,
            far_n=viewpoint_camera.zfar,
            depth_range=viewpoint_camera.depth_range.item(),
            use_view_dependent_phase=pc.use_view_dependent_phase,
            optimize_phase_offset=opt.optimize_phase_offset,
            optimize_dc_offset=opt.optimize_dc_offset,
        )
        rasterizer_tof = GaussianRasterizer(raster_settings=raster_settings_tof_view)
    else:
        rasterizer_tof = GaussianRasterizer(raster_settings=raster_settings_color_view)

    motion_mask = pc.get_motion_mask
    means3D = torch.zeros((pc.get_xyz.shape), device=pc.get_xyz.device)
    means2D = torch.zeros((screenspace_points.shape), device=screenspace_points.device)
    opacity = torch.zeros((pc.get_opacity.shape), device=pc.get_opacity.device)
    scales = torch.zeros((pc.get_scaling.shape), device=pc.get_scaling.device)
    rotations = torch.zeros((pc.get_rotation.shape), device=pc.get_rotation.device)
    shs = torch.zeros((pc.get_features_color.shape), device=pc.get_features_color.device)
    shs_p = torch.zeros((pc.get_features_phasor.shape), device=pc.get_features_phasor.device)

    if "static" in render_regions:
        means3D[~motion_mask] = pc.get_xyz[~motion_mask]
        means2D[~motion_mask] = screenspace_points[~motion_mask]
        opacity[~motion_mask] = pc.get_opacity[~motion_mask]
        scales[~motion_mask] = pc.get_scaling[~motion_mask]
        rotations[~motion_mask] = pc.get_rotation[~motion_mask]
        shs[~motion_mask] = pc.get_features_color[~motion_mask]
        shs_p[~motion_mask] = pc.get_features_phasor[~motion_mask]
    if "dynamic" in render_regions:
        means3D[motion_mask] = pc.get_xyz[motion_mask] + d_xyz
        means2D[motion_mask] = screenspace_points[motion_mask]
        opacity[motion_mask] = pc.get_opacity[motion_mask]
        scales[motion_mask] = pc.get_scaling[motion_mask]
        rotations[motion_mask] = pc.rotation_activation(pc._rotation[motion_mask] + d_rot)
        shs[motion_mask] = pc.get_features_color[motion_mask] + d_sh
        shs_p[motion_mask] = pc.get_features_phasor[motion_mask] + d_sh_p

    rendered_image, _, rendered_depth_color, _, rendered_acc_color, _, depth_distortion_color, _, _, _, _ = rasterizer_color( #
        means3D = means3D,
        means2D = means2D,
        shs = shs, shs_p = shs_p,
        colors_precomp = None, phasors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    
    _, rendered_phasor, rendered_depth_tof, _, rendered_acc_tof, _, depth_distortion_tof, _, pixels, distribution_tof,  radii_tof = rasterizer_tof( #
        means3D = means3D,
        means2D = means2D,
        shs = shs, shs_p = shs_p,
        colors_precomp = None, phasors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        phase_offset = pc.get_phase_offset if opt.optimize_phase_offset else viewpoint_camera.phase_offset.item(),
        dc_offset = pc.get_dc_offset if opt.optimize_dc_offset else viewpoint_camera.dc_offset.item()
        )
    
    return {"render": rendered_image, "render_phasor": rendered_phasor,
            "render_depth": rendered_depth_tof, "render_depth_color": rendered_depth_color,
            "render_acc": rendered_acc_tof, "render_acc_color": rendered_acc_color,
            "depth_distortion": depth_distortion_tof, "depth_distortion_color": depth_distortion_color, 
            "viewspace_points": screenspace_points, # "visibility_filter" : torch.logical_or((radii_tof > 0), (radii_color > 0)),
            "visibility_filter" : radii_tof > 0, # "radii": torch.max(radii_tof, radii_color)
            "radii" : radii_tof,
            "distribution_tof" : distribution_tof,
            "pixels" : pixels
            }

def render_flow(viewpoint_camera : ToFCamera, pc : GaussianModel, d_xyz, d_rot, flow3d, bg_color : torch.Tensor, render_regions=["static", "dynamic"]):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    raster_settings_tof_view = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.tof_image_height),
        image_width=int(viewpoint_camera.tof_image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx_tof * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy_tof * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform_tof,
        projmatrix=viewpoint_camera.full_proj_transform_tof,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center_tof, 
        prefiltered=False,
        debug=False,
        near_n=viewpoint_camera.znear,
        far_n=viewpoint_camera.zfar,
        depth_range=viewpoint_camera.depth_range.item(),
        use_view_dependent_phase=pc.use_view_dependent_phase,
        optimize_phase_offset=False,
        optimize_dc_offset=False,
    )
    rasterizer_tof = GaussianRasterizer(raster_settings=raster_settings_tof_view)

    motion_mask = pc.get_motion_mask
    means3D = torch.zeros((pc.get_xyz.shape), device=pc.get_xyz.device)
    means2D = torch.zeros((screenspace_points.shape), device=screenspace_points.device)
    opacity = torch.zeros((pc.get_opacity.shape), device=pc.get_opacity.device)
    scales = torch.zeros((pc.get_scaling.shape), device=pc.get_scaling.device)
    rotations = torch.zeros((pc.get_rotation.shape), device=pc.get_rotation.device)
    flow3d_ = torch.zeros((pc.get_xyz.shape), device=pc.get_xyz.device)

    if "static" in render_regions:
        means3D[~motion_mask] = pc.get_xyz[~motion_mask]
        means2D[~motion_mask] = screenspace_points[~motion_mask]
        opacity[~motion_mask] = pc.get_opacity[~motion_mask]
        scales[~motion_mask] = pc.get_scaling[~motion_mask]
        rotations[~motion_mask] = pc.get_rotation[~motion_mask]
        flow3d_[~motion_mask] = torch.zeros_like(pc.get_xyz[~motion_mask]).cuda()
    if "dynamic" in render_regions:
        means3D[motion_mask] = pc.get_xyz[motion_mask] + d_xyz
        means2D[motion_mask] = screenspace_points[motion_mask]
        opacity[motion_mask] = pc.get_opacity[motion_mask]
        scales[motion_mask] = pc.get_scaling[motion_mask]
        rotations[motion_mask] = pc.rotation_activation(pc._rotation[motion_mask] + d_rot)
        flow3d_[motion_mask] = flow3d

    rendered_3dflow, _, _, _, _, _, _, _, _, _, _ = rasterizer_tof(
        means3D = means3D.detach(),
        means2D = means2D.detach(),
        shs = None, shs_p = None,
        colors_precomp = flow3d_, phasors_precomp = None,
        opacities = opacity.detach(),
        scales = scales.detach(),
        rotations = rotations.detach(),
        cov3D_precomp = None)
    
    return {"render_flow": rendered_3dflow}

def render_eval(viewpoint_camera : ToFCamera, pc : GaussianModel, d_xyz, d_rot, d_sh, d_sh_p, bg_color : torch.Tensor, render_regions=["static", "dynamic"], tof=False, optimized_phase_offset=False, optimized_dc_offset=False):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if tof:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.tof_image_height),
            image_width=int(viewpoint_camera.tof_image_width),
            tanfovx=math.tan(viewpoint_camera.FoVx_tof * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy_tof * 0.5),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform_tof,
            projmatrix=viewpoint_camera.full_proj_transform_tof,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center, 
            prefiltered=False,
            debug=False,
            near_n=viewpoint_camera.znear,
            far_n=viewpoint_camera.zfar,
            depth_range=viewpoint_camera.depth_range.item(),
            use_view_dependent_phase=pc.use_view_dependent_phase,
            optimize_phase_offset=optimized_phase_offset,
            optimize_dc_offset=optimized_dc_offset,
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center, 
            prefiltered=False,
            debug=False,
            near_n=viewpoint_camera.znear,
            far_n=viewpoint_camera.zfar,
            depth_range=viewpoint_camera.depth_range.item(),
            use_view_dependent_phase=pc.use_view_dependent_phase,
            optimize_phase_offset=optimized_phase_offset,
            optimize_dc_offset=optimized_dc_offset,
        )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    motion_mask = pc.get_motion_mask
    means3D = torch.zeros((pc.get_xyz.shape), device=pc.get_xyz.device)
    means2D = torch.zeros((screenspace_points.shape), device=screenspace_points.device)
    opacity = torch.zeros((pc.get_opacity.shape), device=pc.get_opacity.device)
    scales = torch.zeros((pc.get_scaling.shape), device=pc.get_scaling.device)
    rotations = torch.zeros((pc.get_rotation.shape), device=pc.get_rotation.device)
    shs = torch.zeros((pc.get_features_color.shape), device=pc.get_features_color.device)
    shs_p = torch.zeros((pc.get_features_phasor.shape), device=pc.get_features_phasor.device)

    if "static" in render_regions:
        means3D[~motion_mask] = pc.get_xyz[~motion_mask]
        means2D[~motion_mask] = screenspace_points[~motion_mask]
        opacity[~motion_mask] = pc.get_opacity[~motion_mask]
        scales[~motion_mask] = pc.get_scaling[~motion_mask]
        rotations[~motion_mask] = pc.get_rotation[~motion_mask]
        shs[~motion_mask] = pc.get_features_color[~motion_mask]
        shs_p[~motion_mask] = pc.get_features_phasor[~motion_mask]
    if "dynamic" in render_regions:
        means3D[motion_mask] = pc.get_xyz[motion_mask] + d_xyz
        means2D[motion_mask] = screenspace_points[motion_mask]
        opacity[motion_mask] = pc.get_opacity[motion_mask]
        scales[motion_mask] = pc.get_scaling[motion_mask]
        rotations[motion_mask] = pc.rotation_activation(pc._rotation[motion_mask] + d_rot)
        shs[motion_mask] = pc.get_features_color[motion_mask] + d_sh
        shs_p[motion_mask] = pc.get_features_phasor[motion_mask] + d_sh_p


    rendered_image, rendered_phasor, rendered_depth, _, rendered_acc, _, rendered_dd, _, _, distribution, radii = rasterizer( #
        means3D = means3D,
        means2D = means2D,
        shs = shs, shs_p = shs_p, # Either may be None, remember to deal with this in the rasterizer
        colors_precomp = None, phasors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        phase_offset = pc.get_phase_offset if optimized_phase_offset else viewpoint_camera.phase_offset.item(),
        dc_offset = pc.get_dc_offset if optimized_dc_offset else viewpoint_camera.dc_offset.item()
        )
    
    return {"render": rendered_image, "render_phasor": rendered_phasor, "render_depth": rendered_depth, "render_acc": rendered_acc,  "render_dd": rendered_dd, "distribution": distribution,
            "viewspace_points": screenspace_points, # "visibility_filter" : torch.logical_or((radii_tof > 0), (radii_color > 0)),
            "visibility_filter" : radii > 0, # "radii": torch.max(radii_tof, radii_color)
            "radii" : radii}
