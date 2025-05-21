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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M, int M_p,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs, const float* shs_p,
		bool* clamped, bool* clamped_p,
		const float* cov3D_precomp,
		const float* colors_precomp, const float* phasors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths, float* zs_ndc,
		float* cov3Ds,
		float* rgb, float* real_img_amp, 
		float* dists_to_light, 
		// float* real_img_no_sh, float* real_img_sh_dc, float* real_img_sh_rest, 
		float* phase_amplitude_from_sh, 
		float4* conic_opacity, float3* normal,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		float near_n, float far_n, float depth_range, bool use_view_dependent_phase = false, float phase_offset = 0.0, float dc_offset = 0.0
		);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features, const float* features_p,
		const float* dists_to_light,
		const float4* conic_opacity, const float3* normal, const float* dists_to_light_ndc,
		float* final_T,
		float* final_alpha_total,
		float* final_DD_D, float* final_DD_D2,
		float* final_AD_D, float* final_AD_D2,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color, float* out_phasor, 
		float* out_depth, float* out_normal, float* out_acc, 
		float* out_entropy, float* out_depth_distortion, float* out_amp_distortion, 
		float* pixels,
		float* out_distribution, 
		bool debug
		);
}


#endif