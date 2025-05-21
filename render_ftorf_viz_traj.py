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
from os import makedirs

from utils.general_utils import safe_state
from tqdm import tqdm

import json
from argparse import ArgumentParser
from types import SimpleNamespace

import torch
from scene import Scene
from gaussian_renderer import render_eval, GaussianModel

import imageio
from scene.torf_utils import *
from matplotlib import cm
from utils.graphics_utils import phasor2real_img_amp
import conf
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip, ColorClip, ImageClip, ImageSequenceClip
from matplotlib.collections import LineCollection
from PIL import Image
import matplotlib
matplotlib.use('Agg')

type_names = ["cos", "-cos", "sin", "-sin"]

def generate_color_array(N):
    np.random.seed(42)    
    r = np.random.randint(0, 128, N)
    g = np.random.randint(128, 256, N)
    b = np.random.randint(128, 256, N)
    return np.stack([r, g, b], axis=1).astype(np.uint8)

def is_nearby_used(x, y, used_pixels, radius):
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if (x + dx, y + dy) in used_pixels:
                return True
    return False

def interpolate_trajectory(traj, max_dist=3):
    """
    Linearly interpolate trajectory points if they are too far apart.
    
    :param traj: (N, 2) NumPy array of trajectory points
    :param max_dist: Maximum allowed distance between consecutive points
    :return: Interpolated trajectory as (M, 2) NumPy array
    """
    new_traj = [traj[0]]
    for i in range(1, len(traj)):
        dist = np.linalg.norm(traj[i] - traj[i - 1])
        if dist > max_dist:
            num_steps = int(np.ceil(dist / max_dist))
            interpolated_points = np.linspace(traj[i - 1], traj[i], num_steps, endpoint=False)
            new_traj.extend(interpolated_points[1:])
        new_traj.append(traj[i])
    return np.array(new_traj)

def draw_faded_trajectories(background, trajectories, colors, output_path, max_length=30):
    """
    Draws cartoon-style motion trajectories with tapered width, fading, and optional curvature.

    :param background: Depth map or base image (HxWx3 NumPy array)
    :param trajectories: Dictionary of (N, 2) arrays representing trajectories
    :param colors: Dictionary of (3,) tuples (RGB) for each trajectory
    :param output_path: Path to save the final image
    :param max_length: Maximum visible length of trajectory before fading
    """
    fig, ax = plt.subplots(figsize=(background.shape[1] / 25, background.shape[0] / 25), dpi=150)
    ax.imshow(background, cmap="magma")
    
    for k in trajectories.keys():
        traj = np.array(trajectories[k])
        color = np.array(colors[k]) / 255.0  # Normalize to range [0,1]
        
        if len(traj) > 1:
            traj = interpolate_trajectory(traj)  # Apply interpolation
            visible_traj = traj[-max_length:]  # Keep only recent motion history
            
            # Stronger nonlinear fading effect for alpha and width
            alpha_fade = np.linspace(0, 1, len(visible_traj) - 1)
            alphas = np.power(alpha_fade, 2) * 0.9 + 0.1  # More aggressive fade-out
            widths = np.power(alpha_fade, 2) * 5.0 + 1.0  # Nonlinear width change
            
            segments = [
                [visible_traj[i], visible_traj[i + 1]] for i in range(len(visible_traj) - 1)
            ]
            
            line_colors = [(color[0], color[1], color[2], alphas[i]) for i in range(len(alphas))]
            
            lc = LineCollection(segments, colors=line_colors, linewidths=widths, antialiased=True)
            ax.add_collection(lc)
            ax.plot(visible_traj[-1][0], visible_traj[-1][1], 'o', markersize=3, color=(color[0], color[1], color[2], alphas[-1]))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)

def save_input(model_path, save_folder, views, fps, scene_name, scene_type="torf", tof_permutation=np.array([0,1,2,3]), scale_amp=False, baseline_start_fid=0, baseline_end_fid=60):
    sorted_views = sorted(views, key=lambda x: x.frame_id)
    gt_reals, gt_imags, gt_amps, gt_quads, input_depths, gt_depths, gt_colors = [], [], [], [], [], [], []
    znear, zfar, has_gt = compute_bounds(scene_name)
    for view in sorted_views:
        gt_color = view.original_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0)
        gt_colors.append(gt_color)
        if scene_type == "ftorf":
            quad_default_order = view.original_tofQuad_im[tof_permutation].cpu().detach().numpy().transpose(1, 2, 0)
            tof_from_quad = np.stack([quad_default_order[:, :, 0] - quad_default_order[:, :, 1], quad_default_order[:, :, 2]-quad_default_order[:, :, 3]], axis=-1)
            amp_from_quad = np.sqrt(tof_from_quad[:, :, 0]**2+tof_from_quad[:, :, 1]**2)[..., np.newaxis]
            tof_from_quad = np.concatenate([tof_from_quad, amp_from_quad], axis=-1)
            quad_time_order = view.original_tofQuad_im.cpu().detach().numpy().transpose(1, 2, 0)
            gt_quads.append(quad_time_order[:, :, view.frame_id%4])
            
            gt_real, gt_imag, gt_amp = phasor2real_img_amp(tof_from_quad)
            gt_reals.append(gt_real), gt_imags.append(gt_imag), gt_amps.append(gt_amp)
            if has_gt:
                gt_depth = view.original_distance_image.cpu().detach().numpy()[0]
                gt_depths.append(gt_depth)
            input_depth = depth_from_tof(tof_from_quad, depth_range=view.depth_range, phase_offset=view.phase_offset)[:, :, 0]
            input_depths.append(input_depth)
        else:
            gt_real, gt_imag, gt_amp = phasor2real_img_amp(view.original_tof_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0))
            gt_reals.append(gt_real), gt_imags.append(gt_imag), gt_amps.append(gt_amp)
            if has_gt:
                gt_depth = view.original_distance_image.cpu().detach().numpy()[0]
                gt_depths.append(gt_depth)
            input_depth = depth_from_tof(view.original_tof_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0), depth_range=view.depth_range, phase_offset=view.phase_offset)[:, :, 0]
            input_depths.append(input_depth)

    if len(gt_depths) != 0: # synthetic gt depths w/o motion artifacts
        makedirs(os.path.join(model_path, save_folder, f"gt_depth"), exist_ok=True)
        for vid, im in enumerate(gt_depths):
            disp_im = 1 - (im - znear) / (zfar - znear)
            im_to_write = to8b(cm.magma(disp_im))
            imageio.imwrite(os.path.join(model_path, save_folder, "gt_depth", f"{vid:04d}.png"), im_to_write)
            np.save(os.path.join(os.path.join(model_path, save_folder, "gt_depth"), f"{vid:04d}.npy"), im)
        disp_video = 1 - (np.array(gt_depths) - znear) / (zfar - znear)
        video_to_write = to8b(cm.magma(disp_video))[baseline_start_fid:baseline_end_fid][::4]
        imageio.mimwrite(os.path.join(model_path, save_folder, f'gt_depth.mp4'), video_to_write, fps=fps/4, quality=8)

    makedirs(os.path.join(model_path, save_folder, f"depth"), exist_ok=True)
    for vid, im in enumerate(input_depths):
        disp_im = 1 - (im - znear) / (zfar - znear)
        im_to_write = to8b(cm.magma(disp_im))
        imageio.imwrite(os.path.join(model_path, save_folder, "depth", f"{vid:04d}.png"), im_to_write)
        np.save(os.path.join(os.path.join(model_path, save_folder, "depth"), f"{vid:04d}.npy"), im)
    disp_video = 1 - (np.array(input_depths) - znear) / (zfar - znear)
    video_to_write = to8b(cm.magma(disp_video))[baseline_start_fid:baseline_end_fid]
    imageio.mimwrite(os.path.join(model_path, save_folder, f'depth.mp4'), video_to_write, fps=fps, quality=8)
    empty_frame = to8b(np.ones(to8b(cm.magma(disp_video))[0].shape))
    video_to_write_ = np.repeat(to8b(cm.magma(disp_video))[baseline_start_fid:baseline_end_fid][::4], 4, axis=0)
    video_to_write = np.concatenate([[empty_frame] * 3, video_to_write_, [video_to_write_[-1]]], axis=0).astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, save_folder, f'depth_padded.mp4'), video_to_write, fps=fps, quality=8)
    
    gt_quads_im = []
    makedirs(os.path.join(model_path, save_folder, f"quad"), exist_ok=True)
    for vid, im_ in enumerate(gt_quads):
        im = im_
        if has_gt:
            im = im_ - 0.5
        if scale_amp:
            im_to_write = tof_to_image(im / (im + 1))
        else:
            im_to_write = tof_to_image(im)
        gt_quads_im.append(im_to_write)
        imageio.imwrite(os.path.join(model_path, save_folder, "quad", f"{vid:04d}_{type_names[vid%4]}.png"), im_to_write)
    for i in range(4):
        quad_video = np.array(gt_quads_im)
        video_to_write = quad_video[baseline_start_fid:baseline_end_fid][i::4]
        imageio.mimwrite(os.path.join(model_path, save_folder, f'quad_q{i}_{type_names[i]}.mp4'), video_to_write, fps=fps/4, quality=8)
        empty_frame = to8b(np.ones(quad_video[baseline_start_fid:baseline_end_fid][0].shape))
        video_to_write_ = np.repeat(quad_video[baseline_start_fid:baseline_end_fid][i::4], 4, axis=0)
        prefix = np.array([empty_frame] * i) if i > 0 else np.empty((0, *empty_frame.shape), dtype=empty_frame.dtype)
        suffix = np.array([video_to_write_[-1]] * (4-i)) if (4-i) > 0 else np.empty((0, *video_to_write_[-1].shape), dtype=video_to_write_.dtype)
        video_to_write = np.concatenate([prefix, video_to_write_, suffix], axis=0).astype(np.uint8)
        imageio.mimwrite(os.path.join(model_path, save_folder, f'quad_q{i}_{type_names[i]}_padded.mp4'), video_to_write, fps=fps, quality=8)

    return [gt_reals, gt_imags, gt_amps], [znear, zfar], has_gt

def render_set(args, sample_args, model_path, save_folder, iteration, views, time_steps, time_steps_denom, gaussians, fps, bg_map, tof_inverse_permutation, input_phasors, zplanes, save_img=True, save_video=True, optimized_phase_offset=False, optimized_dc_offset=False, scene_type="torf", has_gt=False):
    rendered_channels = {}
    for ch in (["depth_motion_track", "depth", "depth_tof", "quad"] if save_img else []):
        ch_ = ch + (("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")
        if ch == "quad":
            rendered_channels[ch_] = {}
            for i in range(4):
                rendered_channels[ch_][i] = []
        else:
            rendered_channels[ch_] = []
        makedirs(os.path.join(model_path, f"ours_{iteration}", save_folder, ch_), exist_ok=True)
    views = sorted(views, key=lambda x: x.frame_id)
    znear, zfar = zplanes

    height, width = views[0].distance_image_height, views[0].distance_image_width

    # Collect depth imgs and GS flow
    depth_seq = []
    gs_xyz_flow = []
    initial_3D_pos = None
    for vid, t, view in tqdm(zip(list(range(len(time_steps))), time_steps, views), desc="Rendering depth imgs", total=len(time_steps)):
        d_xyz, d_rot, d_sh, d_sh_p = 0.0, 0.0, 0.0, 0.0
        if t/time_steps_denom >= 0 and t/time_steps_denom <= 1:
            if scene_type in ["torf"]:
                d_xyz, d_rot, d_sh, d_sh_p = gaussians.query_dmlp(t/time_steps_denom)
            elif scene_type in ["ftorf"]:
                curr_int_fid = (view.frame_id // 4) * 4
                next_int_fid = (view.frame_id // 4 + 1) * 4
                d_xyz_curr, _, _, _ = gaussians.query_dmlp(curr_int_fid / time_steps_denom)
                d_xyz_next, _, _, _ = gaussians.query_dmlp(next_int_fid / time_steps_denom)
                d_xyz = 0.25 * ((view.frame_id - curr_int_fid) * d_xyz_next + (next_int_fid - view.frame_id) * d_xyz_curr)
                gs_xyz_flow.append((d_xyz_next-d_xyz_curr)*0.25)
        render_pkg = render_eval(view, gaussians, d_xyz, d_rot, d_sh, d_sh_p, bg_map, optimized_phase_offset=optimized_phase_offset, optimized_dc_offset=optimized_dc_offset)
        if t == 0:
            motion_mask = gaussians.get_motion_mask
            means3D = torch.zeros((gaussians.get_xyz.shape), device=gaussians.get_xyz.device)
            means3D[~motion_mask] = gaussians.get_xyz[~motion_mask]
            means3D[motion_mask] = gaussians.get_xyz[motion_mask] + d_xyz
            initial_3D_pos = means3D

        for i in range(4):
            quad_im = render_pkg["render_phasor"][3:][tof_inverse_permutation][i].detach().cpu().numpy()
            if has_gt:
                quad_im = quad_im - 0.5
            if args.quad_scale > 1.0: # Tonemap
                quad_im = quad_im / (1 + quad_im)
            rendered_channels["quad"][i].append(tof_to_image(quad_im))
            imageio.imwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, "quad", f"{vid}_{type_names[i]}.png"), tof_to_image(quad_im))
        
        rendered_depth = render_pkg["render_depth"].cpu().detach().numpy()[0]
        rendered_channels["depth"].append(rendered_depth)
        rendered_disp = 1 - (rendered_depth - znear) / (zfar - znear) 
        depth_img = to8b(cm.magma(rendered_disp))
        depth_seq.append(depth_img[:, :, :3])
        if save_img:
            imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth"), f"{vid:04d}.png"), depth_img)
            # np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth"), f"{vid:04d}.npy"), rendered_depth)

        depth_tof = depth_from_tof(render_pkg["render_phasor"].cpu().detach().numpy().transpose(1, 2, 0), view.depth_range, gaussians.get_phase_offset.detach().cpu().numpy().item() if optimized_phase_offset else view.phase_offset)[:, :, 0]
        rendered_channels["depth_tof"].append(depth_tof)
        rendered_disp_tof = 1 - (depth_tof - znear) / (zfar - znear) 
        if save_img:
            imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_tof"), f"{vid:04d}.png"), to8b(cm.magma(rendered_disp_tof)))
            # np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_tof"), f"{vid:04d}.npy"), depth_tof)

    gs_xyz_flow = torch.stack(gs_xyz_flow, dim=0)

    # Derive GS xyz positions
    all_3D_pos = [initial_3D_pos]
    all_2D_pos = [project_points_subset(initial_3D_pos, views[0])]
    gs_xy_flow = []
    for vid, t, view in tqdm(zip(list(range(len(time_steps))), time_steps, views), desc="Computing Gaussian positions", total=len(time_steps)):
        if vid > 0:
            next_3D_pos = all_3D_pos[vid-1] + gs_xyz_flow[vid-1]
            all_3D_pos.append(next_3D_pos)
            all_2D_pos.append(project_points_subset(next_3D_pos, view))
            gs_xy_flow.append(all_2D_pos[-1]-all_2D_pos[-2])
    gs_xy_flow = torch.stack(gs_xy_flow, dim=0)

    # Sample GS for visualization
    # Large motion GS
    big_motion_threshold = torch.quantile(torch.mean(torch.sum(gs_xyz_flow**2, dim=-1), dim=0), sample_args.big_motion_quantile)
    big_big_motion_threshold = torch.quantile(torch.mean(torch.sum(gs_xyz_flow**2, dim=-1), dim=0), 0.95)
    gs_sample_mask = (torch.mean(torch.sum(gs_xyz_flow**2, dim=-1), dim=0) > big_motion_threshold)
    # Large motion & Near GS
    z_distr_threshold = torch.quantile(torch.mean(torch.stack(all_3D_pos, dim=0)[:, gs_sample_mask, -1], dim=0), sample_args.z_distr_quantile)
    gs_sample_mask *= (torch.mean(torch.stack(all_3D_pos, dim=0)[:, :, -1], dim=0) < z_distr_threshold)
    # Large motion & Near & Non-transparent GS
    opacity_threshold = torch.quantile(gaussians.get_opacity[gs_sample_mask], sample_args.opacity_quantile)
    gs_sample_mask *= (gaussians.get_opacity > opacity_threshold).squeeze()
    # Large motion & Near & Non-transparent & Middle-size GS
    small_size_threshold = torch.quantile(torch.mean(gaussians.get_scaling[gs_sample_mask], dim=-1), sample_args.small_size_quantile)
    big_size_threshold = torch.quantile(torch.mean(gaussians.get_scaling[gs_sample_mask], dim=-1), sample_args.big_size_quantile)
    gs_sample_mask *= ((small_size_threshold < torch.mean(gaussians.get_scaling, dim=-1)) * (torch.mean(gaussians.get_scaling, dim=-1) < big_size_threshold)).squeeze()

    # Draw GS trajectories
    all_2D_pos_ = torch.stack(all_2D_pos, dim=0)[:, gs_sample_mask, :]
    all_colors = generate_color_array(all_2D_pos_.shape[1])
    np.random.shuffle(all_colors)

    trajectories = {}
    colors = {}
    for vid, t, view in tqdm(zip(list(range(len(time_steps))), time_steps, views), desc="Drawing motion trajectories", total=len(time_steps)):
        if "baseball" in model_path and "clipped" in sample_args.motion_track_postfix and (vid < 28 or vid > 44): 
            continue
        used_pixels = set()
        for i in range(all_2D_pos_.shape[1]):
            init_x, init_y = all_2D_pos_[0, i].detach().cpu()
            old_x, old_y = all_2D_pos_[vid-1, i].detach().cpu()
            new_x, new_y = all_2D_pos_[vid, i].detach().cpu()
            init_x, init_y, old_x, old_y, new_x, new_y = float(init_x), float(init_y), float(old_x), float(old_y), float(new_x), float(new_y)

            nearby_radius = 1 if "target" in model_path else 1
            if "speed_test_texture" in model_path or "chair" in model_path:
                if init_y > height // 4 and torch.mean(torch.sum(gs_xyz_flow[:, i]**2, dim=-1), dim=0) < big_big_motion_threshold:
                    continue
                if (init_y < height // 4 and init_x > width // 12) or init_x > width // 4:
                    continue

            filter_height = height if "jacks" not in model_path else height // 2
            filter_width = width if "jacks" not in model_path else width // 7 * 5
            if not (0 < init_x < filter_width and 0 < init_y < filter_height):
                continue

            filter_height = 0 if "fan" not in model_path else height // 2
            filter_width = 0 if "fan" not in model_path else width // 2
            if not (filter_width < init_x < width and filter_height < init_y < height):
                continue

            filter_height = 0 if "baseball" not in model_path else height // 5
            filter_width = 0 if "baseball" not in model_path else width // 3
            if not (filter_width < init_x < width and filter_height < init_y < height) and (vid < 44 and "baseball" in model_path):
                continue

            if is_nearby_used(int(init_x), int(init_y), used_pixels, radius=nearby_radius):
                continue
            
            if vid == 0:
                trajectories[i] = [[init_x, init_y]]
                colors[i] = tuple(map(int, all_colors[i]))
                # cv2.circle(trajectory_map, (init_x, init_y), radius=2, color=tuple(map(int, all_colors[i])), thickness=-1)
            else:            
                if not (0 < new_x < width - 1 and 0 < new_y < height - 1 and 0 < old_x < width and 0 < old_y < height):
                    continue
                if i in trajectories.keys():
                    trajectories[i].append([new_x, new_y])
                else:
                    trajectories[i] = [[new_x, new_y]]
                    colors[i] = tuple(map(int, all_colors[i]))
                # cv2.line(trajectory_map, (old_x, old_y), (new_x, new_y), tuple(map(int, all_colors[i])), thickness=2)
            used_pixels.add((int(init_x), int(init_y)))
        
        # Write images
        output_path = os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_motion_track"+(("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")), f"{vid:04d}.png")
        draw_faded_trajectories(depth_seq[vid], trajectories, colors, output_path)
        rendered_channels["depth_motion_track"+(("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")].append(imageio.imread(output_path))

    # Write videos
    if save_video:
        video_to_write = np.array(rendered_channels["depth_motion_track"+(("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")][sample_args.baseline_start_fid:sample_args.baseline_end_fid])
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_motion_track{(("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")}.mp4'), video_to_write, fps=fps, quality=8)

        video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth"][sample_args.baseline_start_fid:sample_args.baseline_end_fid])) - znear) / (zfar - znear)))
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth.mp4'), video_to_write, fps=fps, quality=8)

        video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth_tof"][sample_args.baseline_start_fid:sample_args.baseline_end_fid])) - znear) / (zfar - znear)))
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_tof.mp4'), video_to_write, fps=fps, quality=8)

        for t in range(4):
            for tofType in range(4): 
                video_to_write = np.array((rendered_channels["quad"][tofType][sample_args.baseline_start_fid:sample_args.baseline_end_fid]))[t::4]
                imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'quad_q{t}_{type_names[tofType]}.mp4'), video_to_write, fps=fps/4, quality=8)

                empty_frame = to8b(np.ones(video_to_write[0].shape))
                prefix = np.array([empty_frame] * t) if t > 0 else np.empty((0, *empty_frame.shape), dtype=empty_frame.dtype)
                suffix = np.array([video_to_write[-1]] * (4-t)) if (4-t) > 0 else np.empty((0, *video_to_write[-1].shape), dtype=video_to_write.dtype)
                video_to_write = np.concatenate([prefix, np.repeat(video_to_write, 4, axis=0), suffix], axis=0).astype(np.uint8)
                imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'quad_q{t}_{type_names[tofType]}_padded.mp4'), video_to_write, fps=fps, quality=8)

            video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth"][sample_args.baseline_start_fid:sample_args.baseline_end_fid][t::4])) - znear) / (zfar - znear)))
            imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_q{t}.mp4'), video_to_write, fps=fps/4, quality=8)
            
            empty_frame = to8b(np.ones(to8b(cm.magma(1 - (np.array((rendered_channels["depth"][sample_args.baseline_start_fid:sample_args.baseline_end_fid][t::4])) - znear) / (zfar - znear)))[0].shape))
            video_to_write_ = np.repeat(to8b(cm.magma(1 - (np.array((rendered_channels["depth"][sample_args.baseline_start_fid:sample_args.baseline_end_fid][t::4])) - znear) / (zfar - znear))), 4, axis=0)
            prefix = np.array([empty_frame] * t) if t > 0 else np.empty((0, *empty_frame.shape), dtype=empty_frame.dtype)
            suffix = np.array([video_to_write_[-1]] * (4-t)) if (4-t) > 0 else np.empty((0, *video_to_write_[-1].shape), dtype=video_to_write_.dtype)
            video_to_write = np.concatenate([prefix, video_to_write_, suffix], axis=0).astype(np.uint8)
            imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_q{t}_padded.mp4'), video_to_write, fps=fps, quality=8)
            
            video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth_tof"][sample_args.baseline_start_fid:sample_args.baseline_end_fid][t::4])) - znear) / (zfar - znear)))
            imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_tof_q{t}.mp4'), video_to_write, fps=fps/4, quality=8)

            video_to_write = np.array(rendered_channels["depth_motion_track"+(("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")][sample_args.baseline_start_fid:sample_args.baseline_end_fid][t::4])
            imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_motion_track{(("_"+sample_args.motion_track_postfix) if sample_args.motion_track_postfix else "")}_q{t}.mp4'), video_to_write, fps=fps/4, quality=8)

    torch.cuda.empty_cache()

def get_video_spec(video_path):
    if os.path.exists(video_path) and ("motion_track" not in video_path):
        clip = VideoFileClip(video_path)
    else:
        for root, dirs, files in os.walk(os.path.dirname(video_path)):
            for file in files:
                if file.endswith(".mp4") and ("motion_track" not in file):
                    clip = VideoFileClip(os.path.join(root, file))
                    break
    if "deformablegs" in video_path:
        clip = clip.subclip(clip.duration-int(clip.duration))
    (width, height), duration = clip.size, clip.duration
    clip.close()
    return width, height, duration

def get_video_item(video_path, annotation, image_path=None, height_offset=0, image_margin=None, font_size=16, margin=True, image_margin_offset=4, baseline_duration=None, placeholder=None, twoRows=False):
    if "tmp" in video_path:
        clip = VideoFileClip(video_path)
        (width, height), duration = clip.size, clip.duration
        clip.close()
    else:
        width, height, duration = get_video_spec(video_path)
    if baseline_duration:
        duration = baseline_duration

    if annotation in ["", "/"]:
        top_margin, bottom_margin, left_margin, right_margin = 10, 10, 10, 10
    else:
        top_margin, bottom_margin, left_margin, right_margin = 22, 10, 10, 10
    if not margin:
        top_margin, bottom_margin, left_margin, right_margin = 0, 0, 0, 0
    if annotation == '/' or "xxx" in video_path and image_path is None:
        if placeholder is None:
            clip = ColorClip(size=(width, height), color=(255, 255, 255), duration=duration).margin(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin, color=(255, 255, 255))
        else:
            clip = ColorClip(size=(width, height), color=(255, 255, 255), duration=duration).margin(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin, color=(255, 255, 255))

            border_thickness = 2  # Thickness of the border in pixels
            border_array = np.zeros((height, width, 4), dtype=np.uint8)
            border_array[:border_thickness, :, :] = [128, 128, 128, 255]  # Top border
            border_array[-border_thickness:, :, :] = [128, 128, 128, 255] # Bottom border
            border_array[:, :border_thickness, :] = [128, 128, 128, 255]  # Left border
            border_array[:, -border_thickness:, :] = [128, 128, 128, 255] # Right border

            border = ImageClip(border_array, ismask=False).set_duration(duration)
            border = border.set_position((left_margin, top_margin))  # Position inside margins

            text1 = TextClip(placeholder, fontsize=40, color="gray", font="DejaVu Sans").set_opacity(0.5).set_duration(duration)            
            text2 = TextClip("quads acquired", fontsize=40, color="gray", font="DejaVu Sans").set_opacity(0.5).set_duration(duration)

            full_width = width + left_margin + right_margin
            full_height = height + top_margin + bottom_margin
            if twoRows:
                text1 = text1.set_position(("center", full_height // 2 - 35))
                text2 = text2.set_position(("center", full_height // 2 + 5))
                clip = CompositeVideoClip([clip, border, text1, text2], size=(full_width, full_height))
            else:
                text1 = text1.set_position(("center", full_height // 2 - 20))
                clip = CompositeVideoClip([clip, border, text1], size=(full_width, full_height))
    else:
        if "motion_track" in video_path:
            clip = VideoFileClip(video_path).resize((width, height)).margin(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin, color=(255, 255, 255)).set_duration(duration)
        elif image_path is not None:
            if image_margin is not None:
                im_width, im_height = Image.open(image_path).size
                clip = ImageClip(image_path).resize((int(im_width / im_height * height) // 2 * 2, height+height_offset)).set_duration(duration)
                if margin:
                    clip = clip.margin(top=image_margin+image_margin_offset, bottom=image_margin, right=10, left=10, color=(255, 255, 255))
            else:
                clip = ImageClip(image_path).resize((40, height+height_offset)).set_duration(duration)
        else:
            clip = VideoFileClip(video_path).margin(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin, color=(255, 255, 255))
            if "deformablegs" in video_path:
                if clip.duration > 4:
                    clip = clip.subclip(clip.duration-4)
                elif clip.duration > 2:
                    clip = clip.subclip(0, 2)
    if annotation not in ["", "/"]:
        txt_clip = TextClip(annotation, font="DejaVu Sans", fontsize=font_size, color="black").set_position((10, -1)).set_duration(clip.duration).set_start(0)
        final_clip = CompositeVideoClip([clip, txt_clip])
    else:
        final_clip = clip
    return final_clip

def get_vline(video_path, line_width, width, offset, position=("center", "top"), baseline_duration=None):
    _, height, duration = get_video_spec(video_path)
    if baseline_duration:
        duration = baseline_duration
    line = ColorClip(size=(line_width, height + offset + 4), color=(0, 0, 0), duration=duration)
    canvas = ColorClip(size=(width, height + offset), color=(255, 255, 255), duration=duration)
    line = line.set_position(position)
    return CompositeVideoClip([canvas, line])

def draw_time_axis_as_image(panel_height, row_height, labels, width=100, line_x=0.8):
    dpi = 100
    figsize = (width / dpi, panel_height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Convert margin from pixels to normalized coordinates
    margin = 10 / panel_height
    line_start = 1 - margin
    line_end = margin

    # Add arrowhead at bottom
    ax.arrow(line_x, 0.97, 0, -0.915, width=0.025, head_width=0.15, head_length=0.05, fc='black', ec='black')
    ax.text(0.15, 0.978, "Time", fontsize=11, fontname="DejaVu Sans", color="black")
    # ax.text(-0.75, 0.978, "Time", fontsize=11, fontname="DejaVu Sans", color="black")

    for i, label in enumerate(labels):
        # Compute y position with margin considered
        y = line_start - ((i + 0.5) * row_height / panel_height) * (line_start - line_end)
        ax.text(line_x - 0.2, y, label, ha="right", va="center", fontsize=16)

    plt.savefig("utils/tmp/coord_arrow.png", format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return Image.open("utils/tmp/coord_arrow.png")

def make_time_axis_video(height, duration, row_height, labels):
    img = draw_time_axis_as_image(height, row_height, labels)
    frame = np.array(img)
    fh, fw = frame.shape[:2]
    return ImageClip(frame).resize((int(fw / fh * height) // 2 * 2, height)).set_duration(duration)

def make_row_label(text, height, duration, font_size=20):
    """Creates a vertically centered label for a row."""
    txt = TextClip(text, fontsize=font_size, font="DejaVu Sans", color="black", method="label").set_duration(duration)
    
    bg = ColorClip(size=(160, height), color=(255, 255, 255)).set_duration(duration)
    return CompositeVideoClip([bg, txt.set_position(("center", "center"))])

def create_website_video_panel(iteration, motion_track_postfix, model_path, fps, input_folder="gt", renders_folder="renders", num_views=30, scene_type="torf", has_gt=False, baseline_duration=None):
    def mp(p): return os.path.join(model_path, p)
    postfix = "_motion_track" + (("_"+motion_track_postfix) if motion_track_postfix else "")

    ################################################################################## supp layout + separate 4x interpolated
    row1, row2 = [], []
    row1 += [
        get_video_item(mp(f"{input_folder}/depth.mp4"), "C-ToF", baseline_duration=baseline_duration),
        get_video_item(mp(f"baselines/warped.mp4"), "2D Flowed", baseline_duration=baseline_duration),
        get_video_item(mp(f"baselines/deformablegs_depth.mp4"), "Deformable GS", baseline_duration=baseline_duration),
    ]
    row2 += [
        get_video_item(mp(f"baselines/torf.mp4"), "TöRF", baseline_duration=baseline_duration),
        get_video_item(mp(f"baselines/full_model_depth.mp4"), "F-TöRF", baseline_duration=baseline_duration),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth_q0.mp4"), "Ours", baseline_duration=baseline_duration),
    ]
    if has_gt:
        row1 += [
            get_video_item(mp(f"{input_folder}/gt_depth.mp4"), "Ground Truth", baseline_duration=baseline_duration),
            get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth.mp4"), "Ours (Depth), 4x Motion-Based Interpolation", baseline_duration=baseline_duration),
        ]
        row2 += [
            get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), "/", baseline_duration=baseline_duration),
            get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth{postfix}.mp4"), "Ours (3D Gaussian Trajectories)", baseline_duration=baseline_duration),
        ]
    else:
        row1 += [
            get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth.mp4"), "Ours, 4x Motion-Based Interpolation", baseline_duration=baseline_duration),
        ]
        row2 += [
            get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth{postfix}.mp4"), "Ours (3D Gaussian Trajectories)", baseline_duration=baseline_duration),
        ]

    ### Vertical line
    for idx, row in enumerate([row1, row2]):
        row.insert(3 + int(has_gt), get_vline(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), 3, 10, offset=34, position=(5,0), baseline_duration=baseline_duration))
    
    panel = clips_array([
        row1, row2, 
    ], bg_color=(255, 255, 255))

    output_path = mp(f'_tmp.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    # Add colorbar
    row1 = [
        get_video_item(mp(f"_tmp.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp.mp4"), "", f"utils/tmp/magma_colorbar.png", height_offset=-24, image_margin=10, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row1, 
    ], bg_color=(255, 255, 255))
    vd_postfix = "_website"
    output_path = mp(f'iteration_{iteration}_video_panel{vd_postfix}.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    os.system(f"ffmpeg -y -i {output_path} -vcodec libx265 -crf 28 -pix_fmt yuv420p -tag:v hvc1 {output_path[:-4]}_compressed.mp4")
    os.remove(output_path)
    os.rename(output_path[:-4] + "_compressed.mp4", output_path)
    print("[video_panel]: saved.")
    os.remove(mp(f'_tmp.mp4'))

    ################################################################################## Raw quads + 4x interpolated
    # padded = "_padded"
    padded = ""
    row1, row2, row3, row4 = [], [], [], []
    quad_names = ["0", "\u03c0/2", "\u03c0", "3\u03c0/2"]
    row1 += [
        get_video_item(mp(f"{input_folder}/quad_q{0}_{type_names[0]}{padded}.mp4"), f"Quad {quad_names[0]}", font_size=20, baseline_duration=baseline_duration),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"Quad {quad_names[1]}", font_size=20, baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"Quad {quad_names[2]}", font_size=20, baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"Quad {quad_names[3]}", font_size=20, baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), "C-ToF Depth", font_size=20, baseline_duration=baseline_duration, placeholder="1 of 4", twoRows=True),
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/quad_q{0}_{type_names[tofType]}{padded}.mp4"), f"Ours (Rendered Quad {quad_names[tofType]})", font_size=20, baseline_duration=baseline_duration) for tofType in range(4)
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth_q{0}{padded}.mp4"), "Ours (Depth)", font_size=20, baseline_duration=baseline_duration),
    ]
    row2 += [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"{input_folder}/quad_q{1}_{type_names[1]}{padded}.mp4"), f"", baseline_duration=baseline_duration),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="2 of 4", twoRows=True),
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/quad_q{1}_{type_names[tofType]}{padded}.mp4"), "", baseline_duration=baseline_duration) for tofType in range(4)
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth_q{1}{padded}.mp4"), "", baseline_duration=baseline_duration),
    ]
    row3 += [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"{input_folder}/quad_q{2}_{type_names[2]}{padded}.mp4"), f"", baseline_duration=baseline_duration),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="3 of 4", twoRows=True),
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/quad_q{2}_{type_names[tofType]}{padded}.mp4"), "", baseline_duration=baseline_duration) for tofType in range(4)
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth_q{2}{padded}.mp4"), "", baseline_duration=baseline_duration),
    ]
    row4 += [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/xxx.mp4"), f"", baseline_duration=baseline_duration, placeholder="Unknown"),
        get_video_item(mp(f"{input_folder}/quad_q{3}_{type_names[3]}{padded}.mp4"), f"", baseline_duration=baseline_duration),
        get_video_item(mp(f"{input_folder}/depth{padded}.mp4"), "", baseline_duration=baseline_duration),
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/quad_q{3}_{type_names[tofType]}{padded}.mp4"), "", baseline_duration=baseline_duration) for tofType in range(4)
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth_q{3}{padded}.mp4"), "", baseline_duration=baseline_duration),
    ]
    
    panel = clips_array([
        row1[:4], row2[:4], row3[:4], row4[:4],
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_ctof_q.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    panel = clips_array([
        row1[4:5], row2[4:5], row3[4:5], row4[4:5],
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_ctof_d.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    panel = clips_array([
        row1[5:-1], row2[5:-1], row3[5:-1], row4[5:-1],
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_q4.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    panel = clips_array([
        row1[-1:], row2[-1:], row3[-1:], row4[-1:],
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_d4.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    # Add colorbar (C-ToF quads)
    row = [
        get_video_item(mp(f"_tmp_ctof_q.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp_ctof_q.mp4"), "", f"utils/tmp/seismic_colorbar_4rows.png", height_offset=-24, image_margin=10, image_margin_offset=14, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row, 
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_ctof_qb.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    # Add colorbar (C-ToF depths)
    row = [
        get_video_item(mp(f"_tmp_ctof_d.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp_ctof_d.mp4"), "", f"utils/tmp/magma_colorbar_4rows.png", height_offset=-24, image_margin=10, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row, 
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_ctof_db.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    # Add colorbar (Rendered quads)
    row = [
        get_video_item(mp(f"_tmp_q4.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp_q4.mp4"), "", f"utils/tmp/seismic_colorbar_4rows.png", height_offset=-24, image_margin=10, image_margin_offset=14, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row, 
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_q4b.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    # Add colorbar (Rendered depths)
    row = [
        get_video_item(mp(f"_tmp_d4.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp_d4.mp4"), "", f"utils/tmp/magma_colorbar_4rows.png", height_offset=-24, image_margin=10, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row, 
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_d4b.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    # Add time axis for C-ToF
    ctof_qb_clip, ctof_db_clip = VideoFileClip(mp(f'_tmp_ctof_qb.mp4')), VideoFileClip(mp(f'_tmp_ctof_db.mp4'))
    labels = ["t₀", "t₀ + Δt", "t₀ + 2Δt", "t₀ + 3Δt"]
    row_height = row1[0].h
    panel_height = ctof_qb_clip.h
    duration = ctof_qb_clip.duration
    time_axis_clip = make_time_axis_video(panel_height, duration, row_height, labels)
    output_path = mp(f'_tmp_ctof_t.mp4')
    time_axis_clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    time_axis_clip = VideoFileClip(mp(f'_tmp_ctof_t.mp4')).margin(left=10, color=(255,255,255))
    final = clips_array([[time_axis_clip, ctof_qb_clip, ctof_db_clip]], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_ctof{padded}.mp4')
    final.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    # Add time axis for Rendered
    tmp_q4b, tmp_d4b = VideoFileClip(mp(f'_tmp_q4b.mp4')), VideoFileClip(mp(f'_tmp_d4b.mp4'))
    labels = ["t₀", "t₀ + Δt", "t₀ + 2Δt", "t₀ + 3Δt"]
    row_height = row1[0].h
    panel_height = tmp_q4b.h
    duration = tmp_q4b.duration
    time_axis_clip = make_time_axis_video(panel_height, duration, row_height, labels)
    output_path = mp(f'_tmp_t.mp4')
    time_axis_clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    time_axis_clip = VideoFileClip(mp(f'_tmp_t.mp4')).margin(left=10, color=(255,255,255))
    final = clips_array([[time_axis_clip, tmp_q4b, tmp_d4b]], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_ours{padded}.mp4')
    final.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')

    ctof_clip, ours_clip = VideoFileClip(mp(f'_tmp_ctof{padded}.mp4')), VideoFileClip(mp(f'_tmp_ours{padded}.mp4'))
    final = clips_array([[
        ctof_clip, 
        #  get_vline(mp(f"_tmp_q4b.mp4"), 8, 10, offset=34, position=(5,0)),
         get_video_item(mp(f"_tmp_ctof{padded}.mp4"), "", f"utils/tmp/ctof2ours_arrow.png", image_margin=0, margin=False, baseline_duration=baseline_duration),
         ours_clip, 
        #  get_vline(mp(f"_tmp_q4b.mp4"), 8, 10, offset=34, position=(5,0)), 
    ]], bg_color=(255, 255, 255))
    vd_postfix = "_website_quads"
    output_path = mp(f'iteration_{iteration}_video_panel{vd_postfix}{padded}.mp4')
    final.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    os.remove(mp(f'_tmp_q4.mp4'))
    os.remove(mp(f'_tmp_q4b.mp4'))
    os.remove(mp(f'_tmp_d4.mp4'))
    os.remove(mp(f'_tmp_d4b.mp4'))
    # os.remove(mp(f'_tmp_ctof{padded}.mp4'))
    os.remove(mp(f'_tmp_t.mp4'))
    os.remove(mp(f'_tmp_ctof_t.mp4'))
    os.remove(mp(f'_tmp_ctof_db.mp4'))
    os.remove(mp(f'_tmp_ctof_qb.mp4'))
    os.remove(mp(f'_tmp_ctof_d.mp4'))
    os.remove(mp(f'_tmp_ctof_q.mp4'))
    os.system(f"ffmpeg -y -i {output_path} -vcodec libx265 -crf 28 -pix_fmt yuv420p -tag:v hvc1 {output_path[:-4]}_compressed.mp4")
    os.remove(output_path)
    os.rename(output_path[:-4] + "_compressed.mp4", output_path)

    ################################################################################## PPT Layout
    row1, row2 = [], []
    quad_names = ["0", "\u03c0/2", "\u03c0", "3\u03c0/2"]

    sample_clip = get_video_item(mp(f"{input_folder}/depth.mp4"), "", font_size=20)
    video_height = sample_clip.h
    video_duration = sample_clip.duration
    label_gt = make_row_label("GT\n(Desynchronized)", video_height, video_duration)
    label_ours = make_row_label("Ours, Rendered\n(Synchronized)", video_height, video_duration)

    row1 = [label_gt] + [
        get_video_item(mp(f"{input_folder}/quad_q{tofType}_{type_names[tofType]}.mp4"), f"Quad {quad_names[tofType]}", font_size=20, baseline_duration=baseline_duration) for tofType in range(4)
    ] + [
        get_video_item(mp(f"{input_folder}/depth.mp4"), "Depth", font_size=20, baseline_duration=baseline_duration),
    ]
    row2 = [label_ours] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/quad_q{0}_{type_names[tofType]}.mp4"), f"", font_size=20, baseline_duration=baseline_duration) for tofType in range(4)
    ] + [
        get_video_item(mp(f"ours_{iteration}/{renders_folder}/depth_q{0}.mp4"), "", font_size=20, baseline_duration=baseline_duration),
    ]

    panel = clips_array([
        row1[:-1], row2[:-1],
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_q2.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    panel = clips_array([
        row1[-1:], row2[-1:],
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_d.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    # Add colorbar
    row1 = [
        get_video_item(mp(f"_tmp_q2.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp_q2.mp4"), "", f"utils/tmp/seismic_colorbar.png", height_offset=-24, image_margin=10, image_margin_offset=14, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row1, 
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_q2b.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    # Add colorbar
    row1 = [
        get_video_item(mp(f"_tmp_d.mp4"), "", margin=False, baseline_duration=baseline_duration),
        get_video_item(mp(f"_tmp_d.mp4"), "", f"utils/tmp/magma_colorbar.png", height_offset=-24, image_margin=10, baseline_duration=baseline_duration),
    ]
    panel = clips_array([
        row1, 
    ], bg_color=(255, 255, 255))
    output_path = mp(f'_tmp_db.mp4')
    panel.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    
    tmp_q2b, panel = VideoFileClip(mp(f'_tmp_q2b.mp4')).margin(left=10, color=(255,255,255)), VideoFileClip(mp(f'_tmp_db.mp4'))
    final = clips_array([
        [tmp_q2b, 
        #  get_vline(mp(f"_tmp_q2b.mp4"), 3, 10, offset=34, position=(5,0), baseline_duration=baseline_duration), 
         panel]
    ], bg_color=(255, 255, 255))
    vd_postfix = "_website_quads_ppt"
    output_path = mp(f'iteration_{iteration}_video_panel{vd_postfix}.mp4')
    final.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    print("[video_panel]: saved.")
    os.remove(mp(f'_tmp_q2.mp4'))
    os.remove(mp(f'_tmp_q2b.mp4'))
    os.remove(mp(f'_tmp_d.mp4'))
    os.remove(mp(f'_tmp_db.mp4'))
    os.system(f"ffmpeg -y -i {output_path} -vcodec libx265 -crf 28 -pix_fmt yuv420p -tag:v hvc1 {output_path[:-4]}_compressed.mp4")
    os.remove(output_path)
    os.rename(output_path[:-4] + "_compressed.mp4", output_path)

def render_sets(rendering_args, iteration, training_args):
    with torch.no_grad():
        gaussians = GaussianModel(training_args)
        scene = Scene(training_args, gaussians, load_iteration=iteration, shuffle=False)

        background = torch.tensor([0 for _ in range(7)], dtype=torch.float32, device="cuda")
        H, W = scene.getTrainCameras()[0].image_height, scene.getTrainCameras()[0].image_width
        bg_map = background.view(7, 1, 1).expand(7, H, W)
        train_cams, test_cams, spiral_cams = scene.getTrainCameras(), scene.getTestCameras(), scene.getSpiralCameras()
        spiral_cams = spiral_cams[::2]

        time_steps_render = [i for i in range(training_args.total_num_views)]
        time_steps_render_denom = (training_args.total_num_views - 1)

        fps_render = len(time_steps_render[rendering_args.baseline_start_fid:rendering_args.baseline_end_fid]) / rendering_args.baseline_duration

        input_phasors, zplanes, has_gt = save_input(rendering_args.model_path, "input", test_cams, fps_render, scene_name=rendering_args.scene_name, scene_type=scene.scene_type, tof_permutation=scene.tof_permutation, scale_amp=(True if training_args.quad_scale > 1.0 else False), baseline_start_fid=rendering_args.baseline_start_fid, baseline_end_fid=rendering_args.baseline_end_fid)
        print("input saved.")

        render_set(training_args, rendering_args, rendering_args.model_path, "renders", scene.loaded_iter, test_cams, time_steps_render, time_steps_render_denom, gaussians, fps_render, bg_map, scene.tof_inverse_permutation, input_phasors, zplanes, optimized_phase_offset=training_args.optimize_phase_offset, optimized_dc_offset=training_args.optimize_dc_offset, scene_type=scene.scene_type, has_gt=has_gt)
        print("renders saved.")

        create_website_video_panel(scene.loaded_iter, rendering_args.motion_track_postfix, training_args.model_path, min(fps_render, fps_render), input_folder="input", renders_folder="renders", num_views=training_args.total_num_views, scene_type=scene.scene_type, has_gt=has_gt, baseline_duration=rendering_args.baseline_duration)
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--iteration", default=60000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--big_motion_quantile", default=0.9, type=float)
    parser.add_argument("--z_distr_quantile", default=0.5, type=float)
    parser.add_argument("--opacity_quantile", default=0.5, type=float)
    parser.add_argument("--small_size_quantile", default=0.1, type=float)
    parser.add_argument("--big_size_quantile", default=0.9, type=float)
    parser.add_argument("--motion_track_postfix", default="", type=str)
    parser.add_argument("--scene_name", default="", type=str)
    parser.add_argument("--baseline_start_fid", default=0, type=int)
    parser.add_argument("--baseline_end_fid", default=60, type=int)
    parser.add_argument("--baseline_duration", default=4.0, type=float)
    rendering_args = parser.parse_args(sys.argv[1:])

    json_file_path = os.path.join(rendering_args.model_path, "cfg_args_full.json")
    with open(json_file_path, 'r') as f:
        json_args = json.load(f)
    training_args = SimpleNamespace(**json_args)
    training_args.quad_scale = 1.0 # baseball, etc
    if rendering_args.scene_name == "arcing_cube":
        training_args.quad_scale = 5.0
    if rendering_args.scene_name == "jacks1":
        training_args.quad_scale = 2.5
    if rendering_args.scene_name == "target1":
        training_args.quad_scale = 10.0

    print("Rendering " + rendering_args.model_path)

    # Initialize system state (RNG)
    safe_state(rendering_args.quiet, training_args.seed)

    render_sets(rendering_args, rendering_args.iteration, training_args)