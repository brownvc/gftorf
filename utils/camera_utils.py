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

from scene.cameras import ToFCamera
from scene.dataset_readers import CameraInfo
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch, NumpytoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(cam_info : CameraInfo):
    if (cam_info.tof_image is None) and (cam_info.distance_image is None) and (cam_info.image is None): # Spiral
        return ToFCamera(uid=cam_info.uid, frame_id=cam_info.frame_id,
                      R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY, fx=cam_info.fx, fy=cam_info.fy, cx=cam_info.cx, cy=cam_info.cy,
                      image_name=None, image=None, width=cam_info.width, height=cam_info.height,
                      R_tof=cam_info.R_tof, T_tof=cam_info.T_tof, FoVx_tof=cam_info.FovX_tof, FoVy_tof=cam_info.FovY_tof, fx_tof=cam_info.fx_tof, fy_tof=cam_info.fy_tof, cx_tof=cam_info.cx_tof, cy_tof=cam_info.cy_tof,
                      tof_image_name=None, tof_image=None, distance_image_name=None, distance_image=None,
                      tof_width=cam_info.tof_width, tof_height=cam_info.tof_height,
                      gt_alpha_mask=None, znear=cam_info.znear, zfar=cam_info.zfar, depth_range=cam_info.depth_range, phase_offset=cam_info.phase_offset)

    tof_gt_image = None
    if cam_info.tof_image is not None:
        resized_image_tof = NumpytoTorch(cam_info.tof_image, cam_info.image.size)
        tof_gt_image = resized_image_tof[:3, ...]

    distance_gt_image = None
    if cam_info.distance_image is not None:
        if isinstance(cam_info.distance_image, np.ndarray):
            resized_image_distance = NumpytoTorch(cam_info.distance_image, cam_info.image.size)
        elif isinstance(cam_info.distance_image, Image.Image):
            resized_image_distance = PILtoTorch(cam_info.distance_image, cam_info.image.size)
        distance_gt_image = resized_image_distance[:1, ...]
    
    resized_image_rgb = PILtoTorch(cam_info.image, cam_info.image.size)
    color_gt_image = resized_image_rgb[:3, ...]

    color_gt_motion_mask, tof_gt_motion_mask = None, None
    if cam_info.color_motion_mask is not None and cam_info.tof_motion_mask is not None:
        resized_rgb_motion_mask = NumpytoTorch(cam_info.color_motion_mask, cam_info.image.size)
        resized_tof_motion_mask = NumpytoTorch(cam_info.tof_motion_mask, cam_info.image.size)
        color_gt_motion_mask = resized_rgb_motion_mask[0]
        tof_gt_motion_mask = resized_tof_motion_mask[0]

    resized_tofQuad0_im, resized_tofQuad1_im, resized_tofQuad2_im, resized_tofQuad3_im = None, None, None, None
    if cam_info.tofQuad0_im is not None:
        resized_tofQuad0_im = NumpytoTorch(cam_info.tofQuad0_im, cam_info.image.size)
        resized_tofQuad1_im = NumpytoTorch(cam_info.tofQuad1_im, cam_info.image.size)
        resized_tofQuad2_im = NumpytoTorch(cam_info.tofQuad2_im, cam_info.image.size)
        resized_tofQuad3_im = NumpytoTorch(cam_info.tofQuad3_im, cam_info.image.size)

    forward_flow, backward_flow = None, None
    if cam_info.forward_flow is not None:
        forward_flow = NumpytoTorch(cam_info.forward_flow, cam_info.image.size)
    if cam_info.backward_flow is not None:
        backward_flow = NumpytoTorch(cam_info.backward_flow, cam_info.image.size)

    return ToFCamera(uid=cam_info.uid, frame_id=cam_info.frame_id,
                  R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY, fx=cam_info.fx, fy=cam_info.fy, cx=cam_info.cx, cy=cam_info.cy,
                  image_name=cam_info.image_name, image=color_gt_image, width=cam_info.width, height=cam_info.height,
                  R_tof=cam_info.R_tof, T_tof=cam_info.T_tof, FoVx_tof=cam_info.FovX_tof, FoVy_tof=cam_info.FovY_tof, fx_tof=cam_info.fx_tof, fy_tof=cam_info.fy_tof, cx_tof=cam_info.cx_tof, cy_tof=cam_info.cy_tof,
                  tof_image_name=cam_info.tof_image_name, tof_image=tof_gt_image, 
                  distance_image_name=cam_info.distance_image_name, distance_image=distance_gt_image,
                  tof_width=cam_info.tof_width, tof_height=cam_info.tof_height,
                  gt_alpha_mask=None, znear=cam_info.znear, zfar=cam_info.zfar, depth_range=cam_info.depth_range, phase_offset=cam_info.phase_offset,
                  color_motion_mask=color_gt_motion_mask, tof_motion_mask=tof_gt_motion_mask, 
                  tofQuad0_im=resized_tofQuad0_im, tofQuad1_im=resized_tofQuad1_im, tofQuad2_im=resized_tofQuad2_im, tofQuad3_im=resized_tofQuad3_im,
                  dc_offset=cam_info.dc_offset,
                  forward_flow=forward_flow, backward_flow=backward_flow)

def cameraList_from_camInfos(cam_infos):
    camera_list = []

    for c in cam_infos:
        camera_list.append(loadCam(c))
    return camera_list

def camera_to_JSON(id, camera : CameraInfo, save_full=False):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    pos = C2W[:3, 3]
    rot = C2W[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }

    if camera.R_tof is not None:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = camera.R_tof.transpose()
        Rt[:3, 3] = camera.T_tof
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        pos = C2W[:3, 3]
        rot = C2W[:3, :3]
        serializable_array_2d = [x.tolist() for x in rot]
        camera_entry = {
            'id' : id,
            'img_name' : camera.tof_image_name,
            'width' : camera.tof_width,
            'height' : camera.tof_height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(camera.FovY_tof, camera.tof_height),
            'fx' : fov2focal(camera.FovX_tof, camera.tof_width)
        }

    if save_full and camera.R_tof is not None:
        Rt_tof = np.zeros((4, 4))
        Rt_tof[:3, :3] = camera.R_tof.transpose()
        Rt_tof[:3, 3] = camera.T_tof
        Rt_tof[3, 3] = 1.0

        W2C_tof = np.linalg.inv(Rt_tof)
        pos_tof = W2C_tof[:3, 3]
        rot_tof = W2C_tof[:3, :3]
        serializable_array_2d_tof = [x.tolist() for x in rot_tof]

        camera_entry.update({
            'fid' : camera.frame_id,
            'tof_img_name' : camera.tof_image_name,
            'tof_width' : camera.tof_width,
            'tof_height' : camera.tof_height,
            'position_tof': pos_tof.tolist(),
            'rotation_tof': serializable_array_2d_tof,
            'fy_tof' : fov2focal(camera.FovY_tof, camera.tof_height),
            'fx_tof' : fov2focal(camera.FovX_tof, camera.tof_width),
            'znear': camera.znear,
            'zfar': camera.zfar,
            'depth_range': camera.depth_range.item(),
            'phase_offset': camera.phase_offset.item(),
        })
    return camera_entry
