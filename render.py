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
from copy import deepcopy

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
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip, ColorClip

def save_input(model_path, save_folder, views, fps, synthetic=False, save_img=True, save_video=True, scene_type="torf", tof_permutation=np.array([0,1,2,3])):
    for ch in ["real", "imag", "amp", "depth", "color"]:
        makedirs(os.path.join(model_path, save_folder, f"{ch}"), exist_ok=True)

    sorted_views = sorted(views, key=lambda x: x.frame_id)
    gt_reals, gt_imags, gt_amps, gt_depths, gt_colors = [], [], [], [], []
    zplanes = []
    for view in sorted_views:
        gt_color = view.original_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0)
        gt_colors.append(gt_color)
        gt_real, gt_imag, gt_amp = phasor2real_img_amp(view.original_tof_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0))
        gt_reals.append(gt_real), gt_imags.append(gt_imag), gt_amps.append(gt_amp)
        if synthetic:
            gt_depth = view.original_distance_image.cpu().detach().numpy()[0]
            gt_depths.append(gt_depth)
        else:
            input_depth = depth_from_tof(view.original_tof_image[0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0), depth_range=view.depth_range, phase_offset=view.phase_offset)[:, :, 0]
            gt_depths.append(input_depth)
        zplanes.append((0.05 * view.depth_range * 0.9, 0.55 * view.depth_range * 1.1))

    # Write ToF
    for channel_seq, channel_name in zip([gt_reals, gt_imags, gt_amps], ["real", "imag", "amp"]):
        os.makedirs(os.path.join(model_path, save_folder, channel_name), exist_ok=True)
        if save_img:
            for vid, im in enumerate(channel_seq):
                im_to_write = to8b(normalize_im_gt(im, channel_seq))
                imageio.imwrite(os.path.join(model_path, save_folder, channel_name, f"{vid:04d}.png"), im_to_write)
        if save_video:
            video_to_write = to8b(normalize_im(channel_seq))
            # imageio.mimwrite(os.path.join(model_path, save_folder, f'{channel_name}.mp4'), video_to_write, fps=fps, codec='png', quality=10, pixelformat="rgb24")
            imageio.mimwrite(os.path.join(model_path, save_folder, f'{channel_name}.mp4'), video_to_write, fps=fps, quality=8)

    # Write depth
    znear, zfar = zplanes[0]
    if save_img:
        for vid, im in enumerate(gt_depths):
            disp_im = 1 - (im - znear) / (zfar - znear)
            im_to_write = to8b(cm.magma(disp_im))
            imageio.imwrite(os.path.join(model_path, save_folder, "depth", f"{vid:04d}.png"), im_to_write)
            np.save(os.path.join(os.path.join(model_path, save_folder, "depth"), f"{vid:04d}.npy"), im)
    if save_video:
        disp_video = 1 - (np.array(gt_depths) - znear) / (zfar - znear)
        video_to_write = to8b(cm.magma(disp_video))
        # imageio.mimwrite(os.path.join(model_path, save_folder, f'depth.mp4'), video_to_write, fps=fps, codec='png', quality=10, pixelformat="rgb24")
        imageio.mimwrite(os.path.join(model_path, save_folder, f'depth.mp4'), video_to_write, fps=fps, quality=8)
    
    # Write color
    if save_img:
        for vid, im in enumerate(gt_colors):
            im_to_write = to8b(im)
            imageio.imwrite(os.path.join(model_path, save_folder, "color", f"{vid:04d}.png"), to8b(im))
            np.save(os.path.join(os.path.join(model_path, save_folder, "color"), f"{vid:04d}.npy"), im)
    if save_video:
        video_to_write = to8b(gt_colors)
        # imageio.mimwrite(os.path.join(model_path, save_folder, f'color.mp4'), video_to_write, fps=fps, codec='png', quality=10, pixelformat="rgb24")
        imageio.mimwrite(os.path.join(model_path, save_folder, f'color.mp4'), video_to_write, fps=fps, quality=8)

    return [gt_reals, gt_imags, gt_amps], [znear, zfar]

def render_set(args, model_path, save_folder, iteration, views, time_steps, time_steps_denom, gaussians, fps, bg_map, input_phasors, zplanes, save_img=True, save_video=True, optimized_phase_offset=False, optimized_dc_offset=False, scene_type="torf"):
    rendered_channels = {}
    for ch in ["real", "imag", "amp", "depth", "color", "depth_tof", "depth_norm"] + (
        ["dd", "tof", "depth_norm_tof_cam", "quad"] if save_img else []):
        rendered_channels[ch] = []
        makedirs(os.path.join(model_path, f"ours_{iteration}", save_folder, ch), exist_ok=True)

    # Write images
    views = sorted(views, key=lambda x: x.frame_id)
    znear, zfar = zplanes
    for vid, t, view in tqdm(zip(list(range(len(time_steps))), time_steps, views), desc="Rendering progress", total=len(time_steps)):
        d_xyz, d_rot, d_sh, d_sh_p = 0.0, 0.0, 0.0, 0.0
        if t/time_steps_denom >= 0 and t/time_steps_denom <= 1:
            if scene_type in ["torf"]:
                d_xyz, d_rot, d_sh, d_sh_p = gaussians.query_dmlp(t/time_steps_denom)
            elif scene_type in ["ftorf"]:
                curr_int_fid = (view.frame_id // 4) * 4
                next_int_fid = (view.frame_id // 4 + 1) * 4
                d_xyz_curr, _, _, _ = gaussians.query_dmlp(curr_int_fid / time_steps_denom)
                if view.frame_id % 4 == 0 or iteration <= args.optimize_sync_iters:
                    d_xyz = d_xyz_curr 
                else:
                    d_xyz_next, _, _, _ = gaussians.query_dmlp(next_int_fid / time_steps_denom)
                    d_xyz = 0.25 * ((view.frame_id - curr_int_fid) * d_xyz_next + (next_int_fid - view.frame_id) * d_xyz_curr)        
        render_pkg = render_eval(view, gaussians, d_xyz, d_rot, d_sh, d_sh_p, bg_map, optimized_phase_offset=optimized_phase_offset, optimized_dc_offset=optimized_dc_offset)
        # render_pkg_static = render_eval(view, gaussians, d_xyz, d_rot, d_sh, d_sh_p, bg_map, render_regions=["static"], optimized_phase_offset=optimized_phase_offset, optimized_dc_offset=optimized_dc_offset)
        # render_pkg_dynamic = render_eval(view, gaussians, d_xyz, d_rot, d_sh, d_sh_p, bg_map, render_regions=["dynamic"], optimized_phase_offset=optimized_phase_offset, optimized_dc_offset=optimized_dc_offset)
        # gaussians_copy = deepcopy(gaussians)
        # gaussians_copy._xyz[gaussians_copy.get_motion_mask] += d_xyz
        # gaussians_copy.save_ply(os.path.join(model_path, "point_cloud", f"iteration_{int(iteration+t+1)}", "point_cloud.ply"), sibr_only=True)

        if save_img:
            render_pkg_tof_cam = render_eval(view, gaussians, d_xyz, 0.0, 0.0, 0.0, bg_map, tof=True, optimized_phase_offset=optimized_phase_offset, optimized_dc_offset=optimized_dc_offset)

        tof_multiplier = 1.0
        if scene_type == "ftorf" and args.use_quad:
            tof_multiplier = 2.0
        real, imag, amp = phasor2real_img_amp(render_pkg["render_phasor"].cpu().detach().numpy().transpose(1, 2, 0) * tof_multiplier)
        rendered_channels["real"].append(real)
        rendered_channels["imag"].append(imag)
        rendered_channels["amp"].append(amp)
        if save_img:
            for ch, ch_im, ch_gt in zip(["real", "imag", "amp"], [real, imag, amp], input_phasors):
                ch_im_normalized = normalize_im_gt(ch_im, ch_gt)
                imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, ch), f"{vid:04d}.png"), to8b(ch_im_normalized))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "tof"), f"{vid:04d}.npy"), render_pkg["render_phasor"].cpu().detach().numpy().transpose(1, 2, 0))
            type_names = ["cos", "-cos", "sin", "-sin"]
            for tofType_i in range(4):
                quad_im =  render_pkg["render_phasor"][3:][tofType_i].detach().cpu().numpy()
                imageio.imwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, "quad", f"{vid}_{type_names[tofType_i]}.png"), to8b(np.abs(quad_im)))

        depth_tof = depth_from_tof(render_pkg["render_phasor"].cpu().detach().numpy().transpose(1, 2, 0), view.depth_range, gaussians.get_phase_offset.detach().cpu().numpy().item() if optimized_phase_offset else view.phase_offset)[:, :, 0]
        rendered_channels["depth_tof"].append(depth_tof)
        rendered_disp_tof = 1 - (depth_tof - znear) / (zfar - znear) 
        if save_img:
            imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_tof"), f"{vid:04d}.png"), to8b(cm.magma(rendered_disp_tof)))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_tof"), f"{vid:04d}.npy"), depth_tof)
    
        rendered_depth = render_pkg["render_depth"].cpu().detach().numpy()[0]
        rendered_channels["depth"].append(rendered_depth)
        rendered_disp = 1 - (rendered_depth - znear) / (zfar - znear) 
        if save_img:
            imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth"), f"{vid:04d}.png"), to8b(cm.magma(rendered_disp)))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth"), f"{vid:04d}.npy"), rendered_depth)

        rendered_depth_norm = render_pkg["render_depth"].cpu().detach().numpy()[0] / render_pkg["render_acc"].cpu().detach().numpy()[0]
        rendered_channels["depth_norm"].append(rendered_depth_norm)
        rendered_disp_norm = 1 - (rendered_depth_norm - znear) / (zfar - znear) 
        if save_img:
            imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_norm"), f"{vid:04d}.png"), to8b(cm.magma(rendered_disp_norm)))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_norm"), f"{vid:04d}.npy"), rendered_depth_norm)

        if save_img:
            rendered_depth_norm_tof_cam = render_pkg_tof_cam["render_depth"].cpu().detach().numpy()[0] / render_pkg_tof_cam["render_acc"].cpu().detach().numpy()[0]
            rendered_channels["depth_norm_tof_cam"].append(rendered_depth_norm_tof_cam)
            rendered_disp_norm_tof_cam = 1 - (rendered_depth_norm_tof_cam - znear) / (zfar - znear) 
            imageio.imwrite(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_norm_tof_cam"), f"{vid:04d}.png"), to8b(cm.magma(rendered_disp_norm_tof_cam)))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "depth_norm_tof_cam"), f"{vid:04d}.npy"), rendered_depth_norm_tof_cam)

        rendered_color = render_pkg["render"].cpu().detach().numpy().transpose(1, 2, 0)
        rendered_channels["color"].append(rendered_color)
        if save_img:
            imageio.imwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, "color", f"{vid:04d}.png"), to8b(rendered_color))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "color"), f"{vid:04d}.npy"), rendered_color)

        rendered_dd = render_pkg["render_dd"].cpu().detach().numpy()[0]
        if save_img:
            rendered_dd_png = normalize_im(rendered_dd)
            imageio.imwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, "dd", f"{vid:04d}.png"), to8b(rendered_dd_png))
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "dd"), f"{vid:04d}.npy"), rendered_dd)

        if save_img and vid == 0:
            makedirs(os.path.join(model_path, f"ours_{iteration}", save_folder, "distribution"), exist_ok=True)
            rendered_distribution = render_pkg["distribution"].cpu().detach().numpy().transpose(1, 2, 0)
            np.save(os.path.join(os.path.join(model_path, f"ours_{iteration}", save_folder, "distribution"), f"{vid:04d}.npy"), rendered_distribution)

    # Write videos
    if save_video:
        video_to_write = to8b(rendered_channels["color"])
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'color.mp4'), video_to_write, fps=fps, quality=8)
    
        for ch, ch_gt in zip(["real", "imag", "amp"], input_phasors):
            video_to_write = to8b(normalize_im_gt(rendered_channels[ch], ch_gt))
            imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'{ch}.mp4'), video_to_write, fps=fps, quality=8)
    
        video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth"])) - znear) / (zfar - znear)))
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth.mp4'), video_to_write, fps=fps, quality=8)

        video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth_tof"])) - znear) / (zfar - znear)))
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_tof.mp4'), video_to_write, fps=fps, quality=8)

        video_to_write = to8b(cm.magma(1 - (np.array((rendered_channels["depth_norm"])) - znear) / (zfar - znear)))
        imageio.mimwrite(os.path.join(model_path, f"ours_{iteration}", save_folder, f'depth_norm.mp4'), video_to_write, fps=fps, quality=8)

    torch.cuda.empty_cache()
        
def get_video_item(video_path, annotation):
    if annotation == '/':
        for root, dirs, files in os.walk(os.path.dirname(video_path)):
            for file in files:
                if file.endswith(".mp4"):
                    clip = VideoFileClip(os.path.join(root, file))
                    break
        (width, height), duration = clip.size, clip.duration
        final_clip = ColorClip(size=(width+20, height+20), color=(255, 255, 255), duration=duration)
    else:
        clip = VideoFileClip(video_path).margin(10, color=(255, 255, 255))
        txt_clip = TextClip(annotation, font="Arial", fontsize=12, color="black").set_position((10, -2)).set_duration(clip.duration).set_start(0)
        final_clip = CompositeVideoClip([clip, txt_clip])
    return final_clip

def create_video_panel(iteration, model_path, fps, input_folder="gt", renders_folder="renders", num_views=30, scene_type="torf"):
    video_types = [
        'depth', 
        # 'depth_norm', 
        'depth_tof', 
        'amp', 
        ]
    
    if scene_type != "ftorf":
        video_types.insert(0, 'color')
        video_types.append('real')
        video_types.append('imag')

    annotations_row_1 = ['Input(Amp)', 'Ours(Amp)', 'Ours_Sprial(Amp)', 'Ours_FreezeFrame_Spiral(Amp)', ]
    annotations_row_2 = ['Input(Real)', 'Ours(Real)', 'Ours_Sprial(Real)', 'Ours_FreezeFrame_Spiral(Real)', ]
    annotations_row_3 = ['Input(Imag)', 'Ours(Imag)', 'Ours_Sprial(Imag)', 'Ours_FreezeFrame_Spiral(Imag)', ]
    annotations_row_4 = ['Input(RGB)', 'Ours(RGB)', 'Ours_Sprial(RGB)', 'Ours_FreezeFrame_Spiral(RGB)', ]

    annotations_row_5 = ['Input(Time-of-Flight Raw Depth)', 'Ours(Depth)', 'Ours_Sprial(Depth)', 'Ours_FreezeFrame_Spiral(Depth)', ]
    # annotations_row_6 = ['/', 'Ours(Depth Normalized)', 'Ours_Sprial(Depth Normalized)', 'Ours_FreezeFrame_Spiral(Depth Normalized)', ]
    annotations_row_7 = ['/', 'Ours(Depth from ToF)', 'Ours_Sprial(Depth from ToF)', 'Ours_FreezeFrame_Spiral(Depth from ToF)', ]
    
    annotations_list = [annotations_row_5, annotations_row_7, annotations_row_1]
    if scene_type != "ftorf":
        annotations_list.insert(0, annotations_row_4)
        annotations_list.append(annotations_row_2)
        annotations_list.append(annotations_row_3)

    final_clips = []
    for v_ty, annotations in zip(video_types, annotations_list):
        vc_row = []
        # Input
        vc_row.append(get_video_item(os.path.join(model_path, input_folder, v_ty+".mp4"), annotations[0]))

        # Renders
        vc_row.append(get_video_item(os.path.join(model_path, f"ours_{iteration}", renders_folder, v_ty+".mp4"), annotations[1]))

        if scene_type == "torf":
            # Renders Spiral
            if num_views > 1:
                vc_row.append(get_video_item(os.path.join(model_path, f"ours_{iteration}", "renders_spiral", v_ty+".mp4"), annotations[2]))

            # FreezeFrame Spiral
            vc_row.append(get_video_item(os.path.join(model_path, f"ours_{iteration}", "freezeframe_spiral", v_ty+".mp4"), annotations[-1]))
            
        # Add to the panel
        final_clips.append(vc_row)

    if scene_type == "ftorf":
        concatenated_rows = [clips_array([row_clips], bg_color=(255, 255, 255)) for row_clips in list(map(list, zip(*final_clips)))]
    else:
        concatenated_rows = [clips_array([row_clips], bg_color=(255, 255, 255)) for row_clips in final_clips]
    final_video = clips_array([[row] for row in concatenated_rows], bg_color=(255, 255, 255))

    output_path = os.path.join(model_path, f'iteration_{iteration}_video_panel.mp4')
    final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    os.system(f"ffmpeg -y -i {output_path} -vcodec libx265 -crf 28 -pix_fmt yuv420p -tag:v hvc1 {output_path[:-4]}_compressed.mp4")
    os.remove(output_path)
    os.rename(output_path[:-4] + "_compressed.mp4", output_path)
    print(f"[video_panel]: saved.")

def get_video_item_baselines(video_path, annotation):
    clip = VideoFileClip(video_path).margin(10, color=(255, 255, 255))
    txt_clip = TextClip(annotation, font="Arial", fontsize=12, color="black").set_position((10, -2)).set_duration(clip.duration).set_start(0)
    return CompositeVideoClip([clip, txt_clip])

def create_video_panel_baselines(iteration, ours_model_path, fps, gt_folder="gt"):
    video_types = ['depth', 'color']
    annotations_row_1 = ['Input(Time-of-Flight Raw Depth)', 'Ours_Sprial(Depth)', 'ToRF(Depth)', 'RGB+Raw_Depth(Depth)', 'RGB_Only(Depth)']
    annotations_row_2 = ['Input(RGB)', 'Ours_Sprial(RGB)', 'ToRF(RGB)', 'RGB+Raw_Depth(RGB)', 'RGB_Only(RGB)']

    final_clips = []
    for v_ty, annotations in zip(video_types, [annotations_row_1, annotations_row_2]):
        vc_row = []
        vc_row.append(get_video_item_baselines(os.path.join(ours_model_path, gt_folder, v_ty+".mp4"), annotations[0]))
        vc_row.append(get_video_item_baselines(os.path.join(ours_model_path, f"ours_{iteration}", "renders_spiral", v_ty+".mp4"), annotations[1]))
        vc_row.append(get_video_item_baselines(os.path.join(ours_model_path, "baselines", f"{v_ty}_torf.mp4"), annotations[2]))
        vc_row.append(get_video_item_baselines(os.path.join(ours_model_path, "baselines", f"{v_ty}_rgb_depth.mp4"), annotations[3]))
        vc_row.append(get_video_item_baselines(os.path.join(ours_model_path, "baselines", f"{v_ty}_rgb_only.mp4"), annotations[4]))
        final_clips.append(vc_row)

    concatenated_rows = [clips_array([row_clips], bg_color=(255, 255, 255)) for row_clips in final_clips]
    final_video = clips_array([[row] for row in concatenated_rows], bg_color=(255, 255, 255))

    output_path = os.path.join(ours_model_path, f'iteration_{iteration}_baselines.mp4')
    final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', bitrate='8000k')
    os.system(f"ffmpeg -y -i {output_path} -vcodec libx265 -crf 28 -pix_fmt yuv420p -tag:v hvc1 {output_path[:-4]}_compressed.mp4")
    os.remove(output_path)
    os.rename(output_path[:-4] + "_compressed.mp4", output_path)
    print(f"[video_panel]: saved.")

def render_sets(rendering_args, iteration, training_args):
    with torch.no_grad():
        gaussians = GaussianModel(training_args)
        scene = Scene(training_args, gaussians, load_iteration=iteration, shuffle=False)

        background = torch.tensor([0 for _ in range(7)], dtype=torch.float32, device="cuda")
        H, W = scene.getTrainCameras()[0].image_height, scene.getTrainCameras()[0].image_width
        bg_map = background.view(7, 1, 1).expand(7, H, W)
        train_cams, test_cams, spiral_cams = scene.getTrainCameras(), scene.getTestCameras(), scene.getSpiralCameras()
        spiral_cams = spiral_cams[::2]

        phase_offset_final = gaussians.get_phase_offset.detach().cpu().numpy().item() if training_args.optimize_phase_offset else train_cams[0].phase_offset.item()
        gs_mean_zdepth = gaussians.get_xyz[:, -1].mean().detach().cpu().numpy().item()
        with open(os.path.join(rendering_args.model_path, "phase_offset_final.json"), 'w') as file:
            json.dump([{
                "phase_offset_final": phase_offset_final,
                "gs_mean_zdepth": gs_mean_zdepth,
            }], file, indent=4)

        time_steps_render = [i for i in range(training_args.total_num_views)]
        time_steps_render_denom = (training_args.total_num_views - 1)
        time_steps_spiral = [i for i in range(len(spiral_cams))]
        time_steps_spiral_denom = float(len(spiral_cams) - 1)
        time_steps_freezeframe = [training_args.total_num_views // 2] * len(spiral_cams)
        time_steps_freezeframe_denom = (training_args.total_num_views - 1)

        fps_render = len(time_steps_render) / 2.4
        fps_spiral = len(time_steps_spiral) / 2.4

        input_phasors, zplanes = save_input(rendering_args.model_path, "input", test_cams, fps_render, scene_type=scene.scene_type)
        print("input saved.")
        
        render_set(training_args, rendering_args.model_path, "renders", scene.loaded_iter, test_cams, time_steps_render, time_steps_render_denom, gaussians, fps_render, bg_map, input_phasors, zplanes, optimized_phase_offset=training_args.optimize_phase_offset, optimized_dc_offset=training_args.optimize_dc_offset, scene_type=scene.scene_type)
        print("renders saved.")

        if scene.scene_type == "torf":
            render_set(training_args, rendering_args.model_path, "renders_spiral", scene.loaded_iter, spiral_cams, time_steps_spiral, time_steps_spiral_denom, gaussians, fps_spiral, bg_map, input_phasors, zplanes, save_img=True, optimized_phase_offset=training_args.optimize_phase_offset, optimized_dc_offset=training_args.optimize_dc_offset)
            print("renders_spiral saved.")
            
            render_set(training_args, rendering_args.model_path, "freezeframe_spiral", scene.loaded_iter, spiral_cams, time_steps_freezeframe, time_steps_freezeframe_denom, gaussians, fps_spiral, bg_map, input_phasors, zplanes, save_img=True, optimized_phase_offset=training_args.optimize_phase_offset, optimized_dc_offset=training_args.optimize_dc_offset)
            print("freezeframe_spiral saved.")

        create_video_panel(scene.loaded_iter, training_args.model_path, min(fps_render, fps_render), input_folder="input", renders_folder="renders", num_views=training_args.total_num_views, scene_type=scene.scene_type)

        if scene.scene_type == "torf" and "pretrained" in rendering_args.model_path:
            create_video_panel_baselines(iteration, rendering_args.model_path, fps=30/2.4, gt_folder="input")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    rendering_args = parser.parse_args(sys.argv[1:])

    json_file_path = os.path.join(rendering_args.model_path, "cfg_args_full.json")
    with open(json_file_path, 'r') as f:
        json_args = json.load(f)
    training_args = SimpleNamespace(**json_args)
    training_args.quad_scale = 1.0 

    print("Rendering " + rendering_args.model_path)

    # Initialize system state (RNG)
    safe_state(rendering_args.quiet, training_args.seed)

    render_sets(rendering_args, rendering_args.iteration, training_args)