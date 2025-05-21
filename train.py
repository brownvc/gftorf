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

import os
import sys
from tqdm import tqdm
import uuid
from random import randint
import json
import numpy as np
import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui, render_flow
from utils.loss_utils import l1_loss, ssim, l2_loss, weighted_l1_loss, weighted_l1_loss_quad, weighted_l2_loss_quad
from utils.general_utils import safe_state
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, save_args
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from scene.torf_utils import *
from utils.sh_utils import *

import imageio
from utils.graphics_utils import phasor2real_img_amp
from matplotlib import cm

def training(dataset : ModelParams, opt : OptimizationParams, pipe : PipelineParams, other_args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, shuffle=dataset.shuffle_frames)
    gaussians.training_setup(opt)
    if other_args.start_checkpoint:
        (model_params, first_iter) = torch.load(other_args.start_checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if pipe.debug or other_args.debug_from > opt.iterations:
        gt_reals, gt_imags, gt_amps, gt_scattering_phases = [], [], [], []
        if scene.getTrainCameras()[0].original_tof_image is not None:
            for view in scene.getTrainCameras():
                gt_real, gt_imag, gt_amp = phasor2real_img_amp(view.original_tof_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0))
                gt_reals.append(gt_real), gt_imags.append(gt_imag), gt_amps.append(gt_amp)
                gt_scattering_phase = gt_amp * (depth_from_tof(view.original_tof_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0), depth_range=view.depth_range, phase_offset=view.phase_offset)[:, :, 0])**2
                gt_scattering_phases.append(gt_scattering_phase)

        folder_list = ["real", "imag", "amp", "scattering_phase", "depth", "dd", "color", "n_map", "phase_depth",] + ([f"quad"] if opt.use_quad else [])
        if scene.scene_type in ["ftorf"]: 
            folder_list += ["forward_flow2d", "backward_flow2d", 
                            # "acc", "acc_static", "acc_dynamic", 
                            "depth_static", "depth_dynamic", "d_xyz_hist"]
        for ch in folder_list:
            os.makedirs(os.path.join(scene.model_path, f"tmp_debug_{ch}"), exist_ok=True)
            if ch in ["real", "imag", "amp", "scattering_phase",
                    #   "depth", 
                      "color", "phase_depth", "quad", "forward_flow2d", "backward_flow2d"]:
                os.makedirs(os.path.join(scene.model_path, f"tmp_debug_{ch}_gt"), exist_ok=True)
            if ch in ["acc", "n_map", "dd", "acc_static", "depth_static", "acc_dynamic", "depth_dynamic", "d_xyz_hist"]:
                continue
            os.makedirs(os.path.join(scene.model_path, f"tmp_debug_{ch}_error"), exist_ok=True)
        os.makedirs(os.path.join(scene.model_path, f"tmp_debug_distance_synthetic"), exist_ok=True)
        os.makedirs(os.path.join(scene.model_path, f"tmp_debug_distance_error_synthetic"), exist_ok=True)
        os.makedirs(os.path.join(scene.model_path, f"tmp_debug_scattering_phase_tof_depth"), exist_ok=True)
        os.makedirs(os.path.join(scene.model_path, f"tmp_debug_scattering_phase_tof_depth_error"), exist_ok=True)
        os.makedirs(os.path.join(scene.model_path, f"tmp_debug_tof_gs_distribution"), exist_ok=True)
    
    if opt.use_quad:
        type_names = ["cos", "-cos", "sin", "-sin"]
    #     os.makedirs(os.path.join(scene.model_path, f"tmp_debug_gt_quad_hist"), exist_ok=True)
    #     os.makedirs(os.path.join(scene.model_path, f"tmp_debug_gt_quad"), exist_ok=True)
    #     for i in range(dataset.total_num_views):
    #         view_cam_ = scene.getTrainCameras()[i]
    #         gt_quad_ = view_cam_.original_tofQuad_im.cuda()
            # plt.ioff()
            # plt.hist(gt_quad_[view_cam_.frame_id%4].detach().cpu().numpy().flatten(), bins=100)
            # plt.xlim(-1.5, 1.5)
            # plt.savefig(os.path.join(scene.model_path, f"tmp_debug_gt_quad_hist", f"{view_cam_.frame_id:04d}.png"))
            # plt.close()
            # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_gt_quad", f"{view_cam_.frame_id:04d}_{type_names[scene.tof_inverse_permutation[view_cam_.frame_id%4]]}.png"), to8b(np.abs(gt_quad_[view_cam_.frame_id%4].detach().cpu().numpy())))

    flow_cameras = {}
    for i in range(dataset.total_num_views):
        if i % 4 == 0:
            flow_cameras[i] = {}
            for c in scene.getTrainCameras():
                if c.frame_id - i == -4:
                    flow_cameras[i]["prev"] = c
                if c.frame_id - i == 4:
                    flow_cameras[i]["next"] = c

    if scene.scene_type == "torf":
        render_regions = ["dynamic"] # All dynamic
    elif scene.scene_type == "ftorf":
        if dataset.init_static_first:
            render_regions = ["static"]
        else:
            render_regions = ["static", "dynamic"]

    for iteration in range(first_iter, opt.iterations + 1):   

        current_rng_state = torch.random.get_rng_state()
        H, W = scene.getTrainCameras()[0].image_height, scene.getTrainCameras()[0].image_width
        if dataset.random_bg_color:
            torch.manual_seed(iteration)
            bg_map = torch.rand(7, H, W, dtype=torch.float32, device="cuda") * 2 - 1
        else:
            background = torch.tensor(dataset.bg_color, dtype=torch.float32, device="cuda")
            bg_map = background.view(7, 1, 1).expand(7, H, W)
        bg_map_flow = torch.tensor([0 for _ in range(7)], dtype=torch.float32, device="cuda").view(7, 1, 1).expand(7, H, W)
        torch.random.set_rng_state(current_rng_state)

        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, 0.0, 0.0, 0.0, 0.0, pipe, opt, bg_map, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        gaussians.update_learning_rate(iteration, opt)
        gaussians.deform_model.update_learning_rate(iteration - opt.warm_up)

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # Pick a random Camera & Render
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        while viewpoint_cam.frame_id < dataset.start_id:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        d_xyz, d_rot, d_sh, d_sh_p = 0.0, 0.0, 0.0, 0.0
        if dataset.dynamic and iteration > opt.warm_up:
            if scene.scene_type in ["torf"]:
                d_xyz, d_rot, d_sh, d_sh_p = gaussians.query_dmlp(viewpoint_cam.frame_id / (dataset.total_num_views - 1))
            elif scene.scene_type in ["ftorf"]:
                curr_int_fid = (viewpoint_cam.frame_id // 4) * 4
                next_int_fid = (viewpoint_cam.frame_id // 4 + 1) * 4
                d_xyz_curr, _, _, _ = gaussians.query_dmlp(curr_int_fid / (dataset.total_num_views - 1))
                if viewpoint_cam.frame_id % 4 == 0 or iteration <= opt.optimize_sync_iters:
                    d_xyz = d_xyz_curr 
                else:
                    d_xyz_next, _, _, _ = gaussians.query_dmlp(next_int_fid / (dataset.total_num_views - 1))
                    d_xyz = 0.25 * ((viewpoint_cam.frame_id - curr_int_fid) * d_xyz_next + (next_int_fid - viewpoint_cam.frame_id) * d_xyz_curr)
                render_regions = ["static", "dynamic"]
        render_pkg = render(viewpoint_cam, gaussians, d_xyz, d_rot, d_sh, d_sh_p, pipe, opt, bg_map, render_regions=render_regions)
        # render_pkg_static = render(viewpoint_cam, gaussians, d_xyz, d_rot, d_sh, d_sh_p, pipe, opt, bg_map, render_regions=["static"])
        # render_pkg_dynamic = render(viewpoint_cam, gaussians, d_xyz, d_rot, d_sh, d_sh_p, pipe, opt, bg_map, render_regions=["dynamic"])
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image, phasor, depth = render_pkg["render"], render_pkg["render_phasor"], render_pkg["render_depth"]
        gt_image, gt_phasor, gt_depth = viewpoint_cam.original_image.cuda(), viewpoint_cam.original_tof_image, viewpoint_cam.original_distance_image
        if scene.scene_type in ["torf", "ftorf"]:
            gt_phasor = gt_phasor.cuda()
            gt_quad = viewpoint_cam.original_tofQuad_im.cuda() if opt.use_quad else None
            phase_depth = depth_from_tof_torch(phasor[:3], viewpoint_cam.depth_range.item(), phase_offset=scene.gaussians.get_phase_offset.detach().cpu().numpy().item() if opt.optimize_phase_offset else viewpoint_cam.phase_offset.item())
            gt_phase_depth = depth_from_tof_torch(gt_phasor[:3], viewpoint_cam.depth_range.item(), phase_offset=scene.gaussians.get_phase_offset.detach().cpu().numpy().item() if opt.optimize_phase_offset else viewpoint_cam.phase_offset.item())
            n_map = torch.div(depth, viewpoint_cam.depth_range.item() / 2.0, rounding_mode='floor').detach()
            tof_multiplier = 1.0
            if scene.scene_type == "ftorf" and opt.use_quad:
                tof_multiplier = 2.0
            real, imag, amp = phasor2real_img_amp(phasor[:3].cpu().detach().numpy().transpose(1, 2, 0) * tof_multiplier)
            gt_real, gt_imag, gt_amp = phasor2real_img_amp(gt_phasor.cpu().detach().numpy().transpose(1, 2, 0))
            gt_scattering_phase = gt_amp * (gt_phase_depth.cpu().detach().numpy()**2)
            scattering_phase = amp * (depth.cpu().detach().numpy()[0]**2)
            scattering_phase_tof_depth = amp * (phase_depth.cpu().detach().numpy()**2)
            scattering_phase_error = np.abs(gt_scattering_phase - scattering_phase)
            scattering_phase_tof_depth_error = np.abs(gt_scattering_phase - scattering_phase_tof_depth)
        loss = torch.tensor(0.0).cuda()
        Ll1, Ll1_p = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Color losses
        if gt_image is not None and opt.lambda_color != 0.0:
            Ll1 = weighted_l1_loss(image, gt_image, 0.01) if opt.use_wl1c else l1_loss(image, gt_image)
            loss += opt.lambda_color * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        # ToF losses
        if scene.scene_type in ["torf", "ftorf"]:
            if opt.use_quad:
                if opt.warm_up < iteration <= opt.optimize_sync_iters:
                    tof_gt = gt_quad[scene.tof_permutation][2].unsqueeze(0) # cos, -cos, sin, -sin
                    tof_rendered = phasor[3:][2].unsqueeze(0)
                    # tof_gt = gt_quad[scene.tof_permutation]#.unsqueeze(0) # cos, -cos, sin, -sin
                    # tof_rendered = phasor[3:]
                    # tof_gt = torch.cat([tof_gt_[0]-tof_gt_[1], tof_gt_[2] - tof_gt_[3]], dim=0)
                    # tof_gt = gt_phasor[:opt.num_phasor_channels]
                    # tof_rendered = torch.stack([tof_rendered_[0]-tof_rendered_[1], tof_rendered_[2] - tof_rendered_[3]], dim=0)
                else:
                    tof_gt = gt_quad[viewpoint_cam.frame_id%4].unsqueeze(0)
                    tof_rendered = phasor[3:][scene.tof_inverse_permutation][viewpoint_cam.frame_id%4].unsqueeze(0)
            else:
                tof_gt = gt_phasor[:opt.num_phasor_channels]
                tof_rendered = phasor[:opt.num_phasor_channels]
            if opt.use_wl1p:
                Ll1_p = weighted_l2_loss_quad(tof_rendered, tof_gt, opt.wl1p_e) if opt.use_quad else weighted_l1_loss(tof_rendered, tof_gt, opt.wl1p_e, opt.num_phasor_channels)
            else:
                Ll1_p = l2_loss(tof_rendered, tof_gt)
            loss = loss + opt.lambda_tof * ((1.0 - opt.lambda_dssim) * Ll1_p + opt.lambda_dssim * (1.0 - ssim(tof_rendered, tof_gt)))
        # Depth losses (only for baselines)
        if opt.lambda_depth != 0.0:
            if scene.scene_type in ["torf", "ftorf"]:
                loss += opt.lambda_depth * ((1.0 - opt.lambda_dssim) * l1_loss(depth, gt_phase_depth.unsqueeze(0)) + opt.lambda_dssim * (1.0 - ssim(depth, gt_phase_depth.unsqueeze(0))))
            else:
                loss += opt.lambda_depth * ((1.0 - opt.lambda_dssim) * l1_loss(render_pkg["render_depth_color"], gt_depth) + opt.lambda_dssim * (1.0 - ssim(render_pkg["render_depth_color"], gt_depth)))

        forward_flow_l2, backward_flow_l2 = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        if dataset.dynamic and iteration > opt.warm_up:
            # Motion reg loss
            if opt.lambda_mlp_reg != 0.0:
                loss += opt.lambda_mlp_reg * (torch.abs(d_xyz).mean())

            # Flow loss
            if scene.scene_type in ["ftorf"] and iteration > opt.flow_loss_iter_start and viewpoint_cam.frame_id % 4 == 0:
                points3d_curr = distance_to_points3d(render_pkg["render_depth"].detach(), viewpoint_cam)
                points2d_curr = project_points(points3d_curr, viewpoint_cam)

                if viewpoint_cam.forward_flow is not None:
                    d_xyz_next, _, _, _ = gaussians.query_dmlp((viewpoint_cam.frame_id+4) / (dataset.total_num_views - 1)) # World space gaussian deformation
                    rendered_forward_flow3d = render_flow(viewpoint_cam, gaussians, d_xyz_curr.detach(), d_rot, d_xyz_next - d_xyz, bg_map_flow)["render_flow"] # Soft scene flow
                    # rendered_forward_flow2d = project_flow(points2d_curr, points3d_curr, rendered_forward_flow3d, flow_cameras[viewpoint_cam.frame_id]["next"]) # Perspectively projected optical flow
                    rendered_forward_flow2d = project_flow(points2d_curr, points3d_curr, rendered_forward_flow3d, viewpoint_cam) # Perspectively projected optical flow
                    forward_flow_l2 += torch.square((rendered_forward_flow2d - viewpoint_cam.forward_flow)).mean()

                if viewpoint_cam.backward_flow is not None:
                    d_xyz_prev, _, _, _ = gaussians.query_dmlp((viewpoint_cam.frame_id-4) / (dataset.total_num_views - 1))
                    rendered_backward_flow3d = render_flow(viewpoint_cam, gaussians, d_xyz_curr.detach(), d_rot, d_xyz_prev - d_xyz, bg_map_flow)["render_flow"]
                    # rendered_backward_flow2d = project_flow(points2d_curr, points3d_curr, rendered_backward_flow3d, flow_cameras[viewpoint_cam.frame_id]["prev"])
                    rendered_backward_flow2d = project_flow(points2d_curr, points3d_curr, rendered_backward_flow3d, viewpoint_cam)
                    backward_flow_l2 += torch.square((rendered_backward_flow2d - viewpoint_cam.backward_flow)).mean()
                    
                loss += opt.lambda_flow * (forward_flow_l2 + backward_flow_l2)# * (1 - iteration / opt.iterations)

        # if iteration > opt.acc_loss_iter_start:
        #     loss += opt.lambda_acc * (1.0 - render_pkg["render_acc"]).mean()
        
        if iteration > opt.dd_loss_iter_start and iteration < opt.dd_loss_iter_end:
            loss += opt.lambda_dd * render_pkg["depth_distortion"].mean()

        # Opacity Entropy loss (for dynamic Gaussians)
        if opt.use_opacity_entropy_loss and iteration > opt.oe_loss_iter_start and iteration < opt.oe_loss_iter_end:
            dynamic_opacities = gaussians.get_opacity[gaussians.get_motion_mask]
            loss += opt.lambda_oe * (-dynamic_opacities * torch.log(dynamic_opacities + 1e-10) - (1 - dynamic_opacities) * torch.log(1 - dynamic_opacities + 1e-10)).mean()

        # Scale loss
        if opt.use_scale_loss and iteration > opt.scale_loss_iter_start and iteration < opt.scale_loss_iter_end and iteration > opt.warm_up:
            vis_scales = gaussians.get_scaling[visibility_filter]
            loss += opt.lambda_scale * ((vis_scales.mean(dim=-1)**2).mean())# + vis_scales.std(dim=-1).mean() * 50.0)

        loss.backward()
        iter_end.record()

        # Save per iteration phasor for optimization debug
        # if pipe.debug and iteration <= opt.iterations and (iteration % 10 in [0] or iteration < 10):
        #     tof_gs_distribution = render_pkg["distribution_tof"].cpu().detach().numpy()[:, ::15, ::15]
        #     np.save(os.path.join(scene.model_path, f"tmp_debug_tof_gs_distribution", f"{iteration:05d}.npy"), tof_gs_distribution)

        if pipe.debug and iteration <= opt.iterations and iteration % other_args.debug_interval in [0]:
            # if iteration > opt.warm_up and iteration % 1000 in [0]:
            #     d_xyz_mag_np = torch.sqrt(torch.sum(d_xyz**2, dim=-1)).detach().cpu().numpy()
            #     plt.ioff()
            #     plt.hist(d_xyz_mag_np, bins=100)
            #     plt.savefig(os.path.join(scene.model_path, f"tmp_debug_d_xyz_hist", f"{iteration:05d}.png"))
            #     plt.close()
            
            rendered_depth = depth.cpu().detach().numpy()[0]
            if scene.scene_type in ["torf", "ftorf"]: 
                disp = 1 - (rendered_depth - 0.05*viewpoint_cam.depth_range*0.9) / (0.55*viewpoint_cam.depth_range*1.1 - 0.05*viewpoint_cam.depth_range*0.9)
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_depth", f"{iteration:05d}.png"), to8b(cm.magma(disp)))

                # rendered_depth_static = render_pkg_static["render_depth"].cpu().detach().numpy()[0]
                # disp_static = 1 - (rendered_depth_static - 0.05*viewpoint_cam.depth_range*0.9) / (0.55*viewpoint_cam.depth_range*1.1 - 0.05*viewpoint_cam.depth_range*0.9)
                # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_depth_static", f"{iteration:05d}.png"), to8b(cm.magma(disp_static)))

                # rendered_depth_dynamic = render_pkg_dynamic["render_depth"].cpu().detach().numpy()[0]
                # disp_dynamic = 1 - (rendered_depth_dynamic - 0.05*viewpoint_cam.depth_range*0.9) / (0.55*viewpoint_cam.depth_range*1.1 - 0.05*viewpoint_cam.depth_range*0.9)
                # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_depth_dynamic", f"{iteration:05d}.png"), to8b(cm.magma(disp_dynamic)))

                tof_multiplier = 1.0
                if scene.scene_type == "ftorf" and opt.use_quad:
                    tof_multiplier = 2.0
                real, imag, amp = phasor2real_img_amp(phasor[:3].cpu().detach().numpy().transpose(1, 2, 0) * tof_multiplier)
                gt_real, gt_imag, gt_amp = phasor2real_img_amp(gt_phasor.cpu().detach().numpy().transpose(1, 2, 0))
                real_error, imag_error, amp_error = np.abs(gt_real - real), np.abs(gt_imag - imag), np.abs(gt_amp - amp)
                real_error, imag_error, amp_error = normalize_im(real_error), normalize_im(imag_error), normalize_im(amp_error), 

                # for ch, im, im_error, gt in zip(["real", "imag", "amp", "scattering_phase", "scattering_phase_tof_depth"], [real, imag, amp, scattering_phase, scattering_phase_tof_depth], [real_error, imag_error, amp_error, scattering_phase_error, scattering_phase_tof_depth_error], [gt_reals, gt_imags, gt_amps, gt_scattering_phases, gt_scattering_phases]):
                for ch, im, im_error, gt in zip(["amp", "scattering_phase", "scattering_phase_tof_depth"], [amp, scattering_phase, scattering_phase_tof_depth], [amp_error, scattering_phase_error, scattering_phase_tof_depth_error], [gt_reals, gt_imags, gt_amps, gt_scattering_phases, gt_scattering_phases]):
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_{ch}", f"{iteration:05d}.png"), to8b(normalize_im_gt(im, gt)))
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_{ch}_error", f"{iteration:05d}.png"), to8b(im_error))
                # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_real_gt", f"{iteration:05d}.png"), to8b(normalize_im_gt(gt_real, gt_reals)))
                # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_imag_gt", f"{iteration:05d}.png"), to8b(normalize_im_gt(gt_imag, gt_imags)))
                # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_amp_gt", f"{iteration:05d}.png"), to8b(normalize_im_gt(gt_amp, gt_amps)))
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_scattering_phase_gt", f"{iteration:05d}.png"), to8b(normalize_im_gt(gt_scattering_phase, gt_scattering_phases)))
            
                gt_depth_ = gt_phase_depth.cpu().detach().numpy()
                rendered_depth_error = np.abs(gt_depth_ - rendered_depth)
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_depth_error", f"{iteration:05d}.png"), to8b(normalize_im(rendered_depth_error)))

                # n_map_ = normalize_im_max(n_map.cpu().numpy()[0])
                # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_n_map", f"{iteration:05d}.png"), to8b(n_map_))

                phase_disp = 1 - (phase_depth.detach().cpu().numpy() - 0.05*viewpoint_cam.depth_range*0.9) / (0.55*viewpoint_cam.depth_range*1.1 - 0.05*viewpoint_cam.depth_range*0.9)
                gt_phase_disp = 1 - (gt_phase_depth.detach().cpu().numpy() - 0.05*viewpoint_cam.depth_range*0.9) / (0.55*viewpoint_cam.depth_range*1.1 - 0.05*viewpoint_cam.depth_range*0.9)
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_phase_depth", f"{iteration:05d}.png"), to8b(cm.magma(phase_disp)))
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_phase_depth_gt", f"{iteration:05d}.png"), to8b(cm.magma(gt_phase_disp)))
            
                phase_depth_error = np.abs(gt_phase_depth.detach().cpu().numpy() - phase_depth.detach().cpu().numpy())
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_phase_depth_error", f"{iteration:05d}.png"), to8b(normalize_im(phase_depth_error)))
            else:
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_depth", f"{iteration:05d}.png"), to8b(normalize_im(rendered_depth)))                

            # if gt_depth is not None:
            #     gt_distance_synthetic_ = viewpoint_cam.original_distance_image.cpu().detach().numpy()[0]
            #     rendered_distance_synthetic_error = np.abs(gt_distance_synthetic_ - rendered_depth)
            #     disp_synthetic_gt = 1 - (gt_distance_synthetic_ - 0.05*viewpoint_cam.depth_range*0.9) / (0.55*viewpoint_cam.depth_range*1.1 - 0.05*viewpoint_cam.depth_range*0.9)
            #     imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_distance_synthetic", f"{iteration:05d}.png"), to8b(cm.magma(disp_synthetic_gt)))
            #     imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_distance_error_synthetic", f"{iteration:05d}.png"), to8b(normalize_im(rendered_distance_synthetic_error)))

            # rendered_acc = render_pkg["render_acc"].cpu().detach().numpy()[0]
            # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_acc", f"{iteration:05d}.png"), to8b(rendered_acc))

            # rendered_acc_static = render_pkg_static["render_acc"].cpu().detach().numpy()[0]
            # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_acc_static", f"{iteration:05d}.png"), to8b(rendered_acc_static))

            # rendered_acc_dynamic = render_pkg_dynamic["render_acc"].cpu().detach().numpy()[0]
            # imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_acc_dynamic", f"{iteration:05d}.png"), to8b(rendered_acc_dynamic))

            if scene.scene_type in ["torf"]:
                rendered_image = render_pkg["render"].cpu().detach().numpy().transpose(1, 2, 0)
                gt_image_ = gt_image.cpu().detach().numpy().transpose(1, 2, 0)
                rendered_image_error = np.abs(gt_image_ - rendered_image)
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_color", f"{iteration:05d}.png"), to8b(rendered_image))
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_color_gt", f"{iteration:05d}.png"), to8b(gt_image_))
                imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_color_error", f"{iteration:05d}.png"), to8b(rendered_image_error))

            rendered_dd = render_pkg["depth_distortion"].cpu().detach().numpy()[0]
            imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_dd", f"{iteration:05d}.png"), to8b(normalize_im(rendered_dd)))

            if opt.use_quad:
                for tofType_i in range(4):
                    quad_im = phasor[3:][tofType_i].detach().cpu().numpy()
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_quad", f"{iteration:05d}_{scene.tof_permutation[tofType_i]}_{type_names[tofType_i]}.png"), to8b(np.abs(quad_im)))
                    if scene.tof_permutation[tofType_i] == viewpoint_cam.frame_id % 4:
                        quad_error = normalize_im(np.abs(quad_im - gt_quad[scene.tof_permutation][tofType_i].cpu().detach().numpy()))
                    else:
                        quad_error = np.zeros_like(quad_im)                    
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_quad_error", f"{iteration:05d}_{scene.tof_permutation[tofType_i]}_{type_names[tofType_i]}.png"), to8b(quad_error))
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_quad_gt", f"{iteration:05d}_{scene.tof_permutation[tofType_i]}_{type_names[tofType_i]}.png"), to8b(np.abs(gt_quad[scene.tof_permutation][tofType_i].cpu().detach().numpy())))

            if dataset.dynamic and iteration > opt.warm_up and scene.scene_type in ["ftorf"] and iteration > opt.flow_loss_iter_start and viewpoint_cam.frame_id % 4 == 0:
                if viewpoint_cam.forward_flow is not None:
                    rendered_forward_flow2d_ = rendered_forward_flow2d.cpu().detach().numpy().transpose(1, 2, 0)
                    rendered_forward_flow2d_gt = viewpoint_cam.forward_flow.cpu().detach().numpy().transpose(1, 2, 0)
                    forward_flow2d_viz = flow_to_image(rendered_forward_flow2d_, rendered_forward_flow2d_gt)
                    forward_flow2d_gt_viz = flow_to_image(rendered_forward_flow2d_gt, rendered_forward_flow2d_gt)
                    forward_flow2d_error_viz = flow_to_image(rendered_forward_flow2d_gt - rendered_forward_flow2d_, rendered_forward_flow2d_gt)
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_forward_flow2d", f"{iteration:05d}.png"), forward_flow2d_viz)
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_forward_flow2d_gt", f"{iteration:05d}.png"), forward_flow2d_gt_viz)
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_forward_flow2d_error", f"{iteration:05d}.png"), forward_flow2d_error_viz)
                if viewpoint_cam.backward_flow is not None:
                    rendered_backward_flow2d_ = rendered_backward_flow2d.cpu().detach().numpy().transpose(1, 2, 0)
                    rendered_backward_flow2d_gt = viewpoint_cam.backward_flow.cpu().detach().numpy().transpose(1, 2, 0)
                    backward_flow2d_viz = flow_to_image(rendered_backward_flow2d_, rendered_backward_flow2d_gt)
                    backward_flow2d_gt_viz = flow_to_image(rendered_backward_flow2d_gt, rendered_backward_flow2d_gt)
                    backward_flow2d_error_viz = flow_to_image(rendered_backward_flow2d_gt - rendered_backward_flow2d_, rendered_backward_flow2d_gt)
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_backward_flow2d", f"{iteration:05d}.png"), backward_flow2d_viz)
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_backward_flow2d_gt", f"{iteration:05d}.png"), backward_flow2d_gt_viz)
                    imageio.imwrite(os.path.join(scene.model_path, f"tmp_debug_backward_flow2d_error", f"{iteration:05d}.png"), backward_flow2d_error_viz)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save.
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', opt.lambda_color * (1.0 - opt.lambda_dssim) * Ll1.item(), iteration)
                if gt_phasor is not None:
                    tb_writer.add_scalar('train_loss_patches/l1_p_loss', opt.lambda_tof * (1.0 - opt.lambda_dssim) * Ll1_p.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/l2_forward_flow_loss', opt.lambda_flow * forward_flow_l2.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/l2_backward_flow_loss', opt.lambda_flow * backward_flow_l2.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                if iteration > opt.dd_loss_iter_start and iteration < opt.dd_loss_iter_end:
                    tb_writer.add_scalar('train_loss_patches/dd_loss', opt.lambda_dd * render_pkg["depth_distortion"].mean().item(), iteration)

                tb_writer.add_scalar('train_loss_patches/mean_gs_scattering_phase', SH2PA(scene.gaussians.get_features_phasor[:, 0, 1]).mean().item(), iteration)
                tb_writer.add_scalar('train_loss_patches/mean_gs_scattering_phase_visible', SH2PA(scene.gaussians.get_features_phasor[:, 0, 1][visibility_filter]).mean().item(), iteration)
                tb_writer.add_scalar('train_loss_patches/mean_scattering_phase', scattering_phase.mean(), iteration)
                tb_writer.add_scalar('train_loss_patches/mean_scattering_phase_tof_depth', scattering_phase_tof_depth.mean(), iteration)

                tb_writer.add_scalar('train_loss_patches/mean_scattering_phase_gt', gt_scattering_phase.mean(), iteration)
                
                tb_writer.add_scalar('train_loss_patches/mean_scattering_phase_error', scattering_phase_error.mean(), iteration)
                tb_writer.add_scalar('train_loss_patches/scattering_phase_tof_depth_error', scattering_phase_tof_depth_error.mean(), iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)

                tb_writer.add_scalar('train_loss_patches/mean_depth_error', torch.abs(depth - gt_depth).mean(), iteration)
                tb_writer.add_scalar('train_loss_patches/mean_tof_depth_error', torch.abs(phase_depth - gt_phase_depth).mean(), iteration)
                tb_writer.add_scalar('train_loss_patches/mean_amp_error', np.abs(amp - gt_amp).mean(), iteration)
            training_report(tb_writer, iteration, dataset, opt, other_args, scene, (pipe, opt, bg_map), render_regions)

            if (iteration in other_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if render_regions == ["static"]:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"], apply_mask=~gaussians.get_motion_mask)
                elif render_regions == ["dynamic"]:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"], apply_mask=gaussians.get_motion_mask)
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"], apply_mask=None)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 10 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.scene_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0:
                    if render_regions == ["static"]:
                        gaussians.reset_opacity(apply_mask=~gaussians.get_motion_mask)
                    elif render_regions == ["dynamic"]:
                        gaussians.reset_opacity(apply_mask=gaussians.get_motion_mask)
                    else:
                        gaussians.reset_opacity()
            else:
                if opt.use_opacity_entropy_loss and iteration % opt.densification_interval == 0:
                    gaussians.prune(opt.min_opacity)

            # Optimizer step
            if iteration < opt.iterations:
                torch.nn.utils.clip_grad_norm_(gaussians.deform_model.deform.parameters(), max_norm=1.0)
                if iteration < opt.densify_until_iter:
                    gaussians.optimizer.step()
                if iteration % opt.opacity_reset_interval > 200 or iteration >= opt.densify_until_iter:
                    gaussians.deform_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.deform_model.optimizer.zero_grad(set_to_none = True)

                if iteration == opt.tof_iters:
                    opt.lambda_color = 1.0
                    opt.opacity_reset_interval = int(opt.opacity_reset_interval / 2)

            if (iteration in other_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path: # Define model_path if not given
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)

    # Tmp for SIBR_viewers
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, dataset_args, opt_args, other_args, scene : Scene, renderArgs, render_regions=["static", "dynamic"]):
    if iteration in other_args.test_iterations:
        torch.cuda.empty_cache()
        validation_configs = [
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            {'name': 'train', 'cameras' : scene.getTrainCameras()}]

        for config in validation_configs:
            l1_test, l1_p_test, l2_p_test, l1_d_test, l2_d_test, l2_d_tof_test = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            psnr_test, psnr_p_test = 0.0, 0.0
            num_val_cams = 0
            for viewpoint in config['cameras']:
                d_xyz, d_rot, d_sh, d_sh_p = 0.0, 0.0, 0.0, 0.0
                if dataset_args.dynamic and iteration > opt_args.warm_up:
                    if scene.scene_type in ["torf"]:
                        d_xyz, d_rot, d_sh, d_sh_p = scene.gaussians.query_dmlp(viewpoint.frame_id / (dataset_args.total_num_views - 1))
                    elif scene.scene_type in ["ftorf"]:
                        last_int_fid = (viewpoint.frame_id // 4) * 4
                        next_int_fid = (viewpoint.frame_id // 4 + 1) * 4
                        d_xyz_last, _, _, _ = scene.gaussians.query_dmlp(last_int_fid / (dataset_args.total_num_views - 1))
                        if viewpoint.frame_id % 4 == 0:
                            d_xyz = d_xyz_last 
                        else:
                            d_xyz_next, _, _, _ = scene.gaussians.query_dmlp(next_int_fid / (dataset_args.total_num_views - 1))
                            d_xyz = 0.25 * ((viewpoint.frame_id - last_int_fid) * d_xyz_next + (next_int_fid - viewpoint.frame_id) * d_xyz_last)
                render_pkg = render(viewpoint, scene.gaussians, d_xyz, d_rot, d_sh, d_sh_p, *renderArgs)

                rendered_image = render_pkg["render"]
                gt_image = viewpoint.original_image.to("cuda")
                l1_test += l1_loss(rendered_image, gt_image).mean().double()
                psnr_test += psnr(rendered_image, gt_image).mean().double()

                if scene.scene_type in ["torf", "ftorf"]:
                    rendered_phasor = render_pkg["render_phasor"]
                    gt_phasor = viewpoint.original_tof_image.cuda()
                    if opt_args.use_quad:
                        gt_quad = viewpoint.original_tofQuad_im.cuda()
                        tof_gt = gt_quad[viewpoint.frame_id%4].unsqueeze(0)
                        tof_rendered = rendered_phasor[3:][scene.tof_inverse_permutation][viewpoint.frame_id%4].unsqueeze(0)
                    else:
                        tof_gt = gt_phasor[:opt_args.num_phasor_channels]
                        tof_rendered = rendered_phasor[:opt_args.num_phasor_channels]
                    l1_p_test += l1_loss(tof_rendered, tof_gt).mean().double()
                    l2_p_test += l2_loss(tof_rendered, tof_gt).mean().double()
                    psnr_p_test += psnr(tof_rendered, tof_gt).mean().double()

                # if scene.scene_type in ["blender", "colmap"]:
                gt_depth = viewpoint.original_distance_image
                # elif scene.scene_type in ["torf", "ftorf"]:
                #     gt_depth = depth_from_tof_torch(gt_phasor, viewpoint.depth_range.item(), 
                #                                     phase_offset=scene.gaussians.get_phase_offset.detach().cpu().numpy().item() if opt_args.optimize_phase_offset else viewpoint.phase_offset.item()).unsqueeze(0)
                if gt_depth is not None:
                    gt_depth = gt_depth.to("cuda")
                    rendered_depth = render_pkg["render_depth"]
                    rendered_depth_tof = depth_from_tof_torch(rendered_phasor, viewpoint.depth_range.item(), 
                                                              phase_offset=scene.gaussians.get_phase_offset.detach().cpu().numpy().item() if opt_args.optimize_phase_offset else viewpoint.phase_offset.item()).unsqueeze(0)
                    l1_d_test += l1_loss(rendered_depth, gt_depth).mean().double()
                    l2_d_test += l2_loss(rendered_depth, gt_depth).mean().double()
                    l2_d_tof_test += l2_loss(rendered_depth_tof, gt_depth).mean().double()

                num_val_cams += 1

            psnr_test /= num_val_cams
            l1_test /= num_val_cams

            psnr_p_test /= num_val_cams
            l1_p_test /= num_val_cams
            l2_p_test /= num_val_cams
            
            l1_d_test /= num_val_cams
            l2_d_test /= num_val_cams
            l2_d_tof_test /= num_val_cams

            if tb_writer:
                print(" L1 {} PSNR {}".format(l1_test, psnr_test), end="")
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
                print(" L1_p {} L2_p {} PSNR_p {}".format(l1_p_test, l2_p_test, psnr_p_test), end="")
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_p_loss', l1_p_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l2_p_loss', l2_p_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_p', psnr_p_test, iteration)

                print(" L1_d {} L2_d {}, L2_d_tof {}".format(l1_d_test, l2_d_test, l2_d_tof_test), end="")
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_d_loss', l1_d_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l2_d_loss', l2_d_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l2_d_tof_loss', l2_d_tof_test, iteration)
            print()
        print(f"\nThere are {torch.sum(~scene.gaussians.get_motion_mask)} static Gaussians and {torch.sum(scene.gaussians.get_motion_mask)} dynamic Gaussians.\n")
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity[render_pkg["visibility_filter"]], iteration)
            tb_writer.add_histogram("scene/dists_histogram", (scene.gaussians.get_xyz**2)[render_pkg["visibility_filter"]].sum(dim=-1).sqrt(), iteration)
            tb_writer.add_histogram("scene/amplitude_histogram", SH2PA(scene.gaussians.get_features_phasor[:, 0, 1][render_pkg["visibility_filter"]]), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', required=True, type=str, help='Path to JSON config file')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--eval_interval", nargs="+", type=int, default=11)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_interval", type=int, default=100)
    with open(parser.parse_args(sys.argv[1:]).config, 'r') as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    # This is a debugging tool for finding non-differentiable operations or those lead to numerical instability
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.test_iterations = [1] + np.linspace(0, args.iterations, args.iterations // 1000 + 1).astype(np.int64).tolist()
    args.save_iterations = [args.iterations//2, args.iterations]

    save_args(args, args.model_path, "cfg_args_full.json")

    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
