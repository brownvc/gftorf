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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixShift

class ToFCamera(nn.Module):
    def __init__(self, uid, frame_id, 
                 R, T, FoVx, FoVy, 
                 image_name, image, 
                 width, height,
                 R_tof, T_tof, FoVx_tof, FoVy_tof, 
                 tof_image_name, tof_image, distance_image_name, distance_image,
                 tof_width, tof_height, 
                 gt_alpha_mask, znear=0.01, zfar=100.0, depth_range=100.0, phase_offset=0.0,
                 color_motion_mask=None, tof_motion_mask=None, 
                 tofQuad0_im=None, tofQuad1_im=None, tofQuad2_im=None, tofQuad3_im=None, dc_offset=np.array(0.0).astype(np.float32),
                 forward_flow=None, backward_flow=None,
                 fx=None, fy=None, cx=None, cy=None, 
                 fx_tof=None, fy_tof=None, cx_tof=None, cy_tof=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(ToFCamera, self).__init__()

        self.uid = uid
        self.frame_id = frame_id

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32).cuda()
        self.image_name = image_name

        self.R_tof = R_tof
        self.T_tof = T_tof
        self.FoVx_tof = FoVx_tof
        self.FoVy_tof = FoVy_tof
        self.fx_tof, self.fy_tof = fx_tof, fy_tof
        self.cx_tof, self.cy_tof = cx_tof, cy_tof
        if fx_tof is not None and fx_tof is not None:
            self.K_tof = torch.tensor([[fx_tof,0,cx_tof],[0,fy_tof,cy_tof],[0,0,1]], dtype=torch.float32).cuda()
        self.tof_image_name = tof_image_name

        self.distance_image_name = distance_image_name
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device).float()
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.original_image = None
            self.image_width, self.image_height = width, height

        if tof_image is not None:
            self.original_tof_image = tof_image.to(self.data_device).float()
            self.tof_image_width = self.original_tof_image.shape[2]
            self.tof_image_height = self.original_tof_image.shape[1]
        else:
            self.original_tof_image = None
            self.tof_image_width, self.tof_image_height = tof_width, tof_height

        if distance_image is not None:
            self.original_distance_image = distance_image.to(self.data_device).float()
            self.distance_image_width = self.original_distance_image.shape[2]
            self.distance_image_height = self.original_distance_image.shape[1]
        else:
            self.original_distance_image = None

        if color_motion_mask is not None and tof_motion_mask is not None:
            seg_color = color_motion_mask.to(self.data_device).float()
            seg_tof = tof_motion_mask.to(self.data_device).float()
            self.seg_color = torch.stack((seg_color, torch.zeros_like(seg_color), 1 - seg_color))
            self.seg_tof = torch.stack((seg_tof, torch.zeros_like(seg_tof), 1 - seg_tof))
        else:
            self.seg_color, self.seg_tof = None, None
            
        if tofQuad0_im is not None and tofQuad1_im is not None and tofQuad2_im is not None and tofQuad3_im is not None:
            self.original_tofQuad_im = torch.cat([tofQuad0_im, tofQuad1_im, tofQuad2_im, tofQuad3_im], dim=0).to(self.data_device).float()
        else:
            self.original_tofQuad_im = None

        self.forward_flow, self.backward_flow = None, None
        if forward_flow is not None:
            self.forward_flow = forward_flow.to(self.data_device).float()
        if backward_flow is not None:
            self.backward_flow = backward_flow.to(self.data_device).float()

        self.zfar = zfar
        self.znear = znear
        self.depth_range = depth_range
        self.phase_offset = phase_offset
        self.dc_offset = dc_offset

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if fx is not None:
            self.projection_matrix = getProjectionMatrixShift(
                znear=self.znear, zfar=self.zfar, focal_x=fx, focal_y=fy, cx=cx, cy=cy, 
                width=width, height=height, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if R_tof is not None:
            self.world_view_transform_tof = torch.tensor(getWorld2View2(R_tof, T_tof, trans, scale)).transpose(0, 1).cuda()
            if fx_tof is not None:
                self.projection_matrix_tof = getProjectionMatrixShift(
                    znear=self.znear, zfar=self.zfar, focal_x=fx_tof, focal_y=fy_tof, cx=cx_tof, cy=cy_tof, 
                    width=tof_width, height=tof_height, fovX=self.FoVx_tof, fovY=self.FoVy_tof).transpose(0,1).cuda()
            else:
                self.projection_matrix_tof = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx_tof, fovY=self.FoVy_tof).transpose(0,1).cuda()
            self.full_proj_transform_tof = (self.world_view_transform_tof.unsqueeze(0).bmm(self.projection_matrix_tof.unsqueeze(0))).squeeze(0)
            self.camera_center_tof = self.world_view_transform_tof.inverse()[3, :3]
        else:
            self.world_view_transform_tof = self.world_view_transform
            self.projection_matrix_tof = self.projection_matrix
            self.full_proj_transform_tof = self.full_proj_transform
            self.camera_center_tof = self.camera_center


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
