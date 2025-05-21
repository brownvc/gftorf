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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present,
			float znear,
			float zfar);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M, int M_p,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs, const float* shs_p,
			const float* colors_precomp, const float* phasors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color, float* out_phasor, 
			float* out_depth, float* out_normal, float* out_acc, float* out_entropy, float* out_depth_distortion, float* out_amp_distortion,
			float* pixels, 
			float* out_distribution,
			int* radii = nullptr,
			bool debug = false,
			float near_n = 0.01,
			float far_n = 100.0,
			float depth_range = 100.0,
			bool use_view_dependent_phase = false,
			float phase_offset = 0.0, float dc_offset = 0.0
			);

		static void backward(
			const int P, int D, int M, int M_p, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs, const float* shs_p,
			const float* colors_precomp, const float* phasors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix, const float* dL_dpix_p, 
			const float* dL_dpix_d, const float* dL_dpix_n, const float* dL_dpix_a, 
			const float* dL_dpix_e, const float* dL_dpix_dd, const float* dL_dpix_ad,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor, float* dL_dphasor, float* dL_ddist_to_light, float* dL_ddist_to_light_ndc,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh, float* dL_dsh_p,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dphase_offset,
			float* dL_ddc_offset,
			bool debug,
			float near_n = 0.01,
			float far_n = 100.0,
			float depth_range = 100.0,
			bool use_view_dependent_phase = false,
			float phase_offset = 0.0, float dc_offset = 0.0
			);
	};
};

#endif