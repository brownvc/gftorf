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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors, const float* real_img_amps, const float* dists_to_light, const float* dists_to_light_ndc,
		const float* final_Ts,
		const float* alpha_totals,
		const float* w_z_total, const float* w_z2_total,
		const float* w_amplitude_total, const float* w_amplitude2_total,
		const uint32_t* n_contrib,
		const float* dL_dpixels, const float* dL_dpixels_p, 
		const float* dL_dpixels_d, const float* dL_dpixels_a, 
		const float* dL_dpixels_e, const float* dL_dpixels_dd, const float* dL_dpixels_ad,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors, float* dL_dphasors, float* dL_ddists_to_light, float* dL_ddists_to_light_ndc);

	void preprocess(
		int P, int D, int M, int M_p,
		const float3* means,
		const int* radii,
		const float* shs, const float* shs_p,
		const bool* clamped, const bool* clamped_p,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor, float* dL_dphasor, float* dL_ddist_to_light, float* dL_ddist_to_light_ndc,
		float* dL_dcov3D,
		float* dL_dsh, float* dL_dsh_p,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot, float* dL_dphase_offset, float* dL_ddc_offset,
		const float* phase_amplitude_from_sh_ptr, const float* dists_to_light,
		float near_n, float far_n, float dist2phase, bool use_view_dependent_phase = false, float phase_offset = 0.0, float dc_offset = 0.0
		);
}

#endif