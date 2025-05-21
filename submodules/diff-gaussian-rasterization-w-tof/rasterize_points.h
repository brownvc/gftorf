/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, 
torch::Tensor, torch::Tensor, // Color, Phasor
torch::Tensor, torch::Tensor, torch::Tensor, // Depth, Normal, Acc
torch::Tensor, torch::Tensor, torch::Tensor, // Entropy, Depth Distortion, Amplitude Distortion
torch::Tensor, // Pixels
torch::Tensor, // Distributions
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors, const torch::Tensor& phasors, // Either could be empty,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh, const torch::Tensor& sh_p, // Either could be empty, use sh.defined() to check if defined
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const float near_n,
	const float far_n,
	const float depth_range,
	const bool use_view_dependent_phase,
	const float phase_offset, const float dc_offset
	);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors, const torch::Tensor& phasors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color, const torch::Tensor& dL_dout_phasor, 
	const torch::Tensor& dL_dout_depth, const torch::Tensor& dL_dout_normal, const torch::Tensor& dL_dout_acc, 
	const torch::Tensor& dL_dout_entropy, const torch::Tensor& dL_dout_depth_distortion, const torch::Tensor& dL_dout_amp_distortion, 
	const torch::Tensor& sh, const torch::Tensor& sh_p,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const float near_n,
	const float far_n,
	const float depth_range,
	const bool use_view_dependent_phase,
	const float phase_offset, const float dc_offset
	);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix,
		float znear, float zfar);