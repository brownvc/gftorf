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
import numpy as np
from typing import NamedTuple
from typing import Optional

class BasicPointCloud(NamedTuple):
    points : np.array
    normals : np.array
    colors : np.array
    phases: Optional[np.array] = None
    amplitudes: Optional[np.array] = None
    seg_colors: Optional[np.array] = None

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixShift(znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    offset_x = cx - (width/2)
    offset_x = (offset_x/focal_x)*znear
    offset_y = cy - (height/2)
    offset_y = (offset_y/focal_y)*znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels): # Focal length to field of view (FoV)
    return 2*math.atan(pixels/(2*focal))

def phasor2real_img_amp(phasor):
    '''
    Phasor = H x W x 3 --> Real, Imag, Amp
    Return: red (positive), blue (negative)
    '''
    real, imaginary = phase2real_img(phasor[:, :, :2])
    return real, imaginary, phasor[:, :, 2]

def phase2real_img(phase):
    real = np.tile((phase[:, :, 0])[:, :, np.newaxis], (1, 1, 3))
    real[:, :, 0][real[:, :, 0] <= 0] = 0.0
    real[:, :, 2][real[:, :, 2] >= 0] = 0.0
    real[:, :, 2] = -real[:, :, 2]
    real[:, :, 1] = 0.0

    imaginary = np.tile((phase[:, :, 1])[:, :, np.newaxis], (1, 1, 3))
    imaginary[:, :, 0][imaginary[:, :, 0] <= 0] = 0.0
    imaginary[:, :, 2][imaginary[:, :, 2] >= 0] = 0.0
    imaginary[:, :, 2] = -imaginary[:, :, 2]
    imaginary[:, :, 1] = 0.0
    return real, imaginary
