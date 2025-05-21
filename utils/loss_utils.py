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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def weighted_l1_loss(network_output, gt, w, num_phasor_channels):
    weight = w + torch.sqrt(torch.sum(network_output**2, dim=0)).detach() # weighted by amp
    return torch.abs((network_output[:num_phasor_channels] - gt[:num_phasor_channels]) / weight).mean()

def weighted_l1_loss_quad(network_output, gt, w):
    weight = w + torch.abs(network_output.detach())
    return torch.abs((network_output - gt) / weight).mean()

def weighted_l2_loss_quad(network_output, gt, w):
    weight = w + torch.abs(network_output.detach())
    return torch.square((network_output - gt) / weight).mean()

def find_knn(xyz, k):
    """
    Arguments:
        xyz: shape (P, 3)
    Returns:
        indices of the k nearest neighbors
    """
    dist = torch.cdist(xyz, xyz, p=2)  # size (p, p)
    sorted_indices = torch.topk(dist, k+1, dim=-1, largest=False)[1]  
    return sorted_indices[:, 1:] # size (p, k)

def rigidity_loss(gaussians, prev_xyz, curr_xyz, curr_knn_indices):
    """
    Arguments:
        prev_xyz: shape (P, 3)
        curr_xyz: shape (P, 3)
        curr_knn_indices: shape (P, k)
    Returns: 
        local rigidity loss (scalar)
    """
    curr_knn_pos = curr_xyz[curr_knn_indices]  # size (p, k, 3)
    prev_knn_pos = prev_xyz[curr_knn_indices]  # size (p, k, 3)
    curr_xyz_repeated = curr_xyz.unsqueeze(1).expand(-1, curr_knn_indices.shape[1], -1)  # size (p, k, 3)
    prev_xyz_repeated = prev_xyz.unsqueeze(1).expand(-1, curr_knn_indices.shape[1], -1)  # size (p, k, 3)
    L_i_j = gaussians.w_i_j * torch.sqrt((((prev_xyz_repeated - prev_knn_pos) - (curr_xyz_repeated - curr_knn_pos)) ** 2).sum(dim=-1) + 1e-16) # size (p, k, 3)
    return L_i_j.mean()

def isometry_loss(gaussians, curr_xyz, curr_knn_indices):
    """
    Arguments:
        curr_xyz: shape (P, 3)
        curr_knn_indices: shape (P, k)
    Returns: 
        global isometry loss (scalar)
    """
    curr_knn_pos = curr_xyz[curr_knn_indices]
    curr_xyz_repeated = curr_xyz.unsqueeze(1).expand(-1, curr_knn_indices.shape[1], -1)  # size (p, k, 3)
    return (gaussians.w_i_j * (torch.sqrt(gaussians.initial_xyz_to_knn_dist + 1e-16) - torch.sqrt(((curr_xyz_repeated - curr_knn_pos) ** 2).sum(dim=-1) + 1e-16))).abs().mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

