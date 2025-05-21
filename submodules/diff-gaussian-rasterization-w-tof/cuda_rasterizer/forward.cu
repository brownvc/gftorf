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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__device__ glm::vec2 computePhasorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs_p, bool* clamped_p) {
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec2* sh_p = ((glm::vec2*)shs_p) + idx * max_coeffs;
	glm::vec2 result_p = SH_C0 * sh_p[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result_p = result_p - SH_C1 * y * sh_p[1] + SH_C1 * z * sh_p[2] - SH_C1 * x * sh_p[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result_p = result_p +
				SH_C2[0] * xy * sh_p[4] +
				SH_C2[1] * yz * sh_p[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh_p[6] +
				SH_C2[3] * xz * sh_p[7] +
				SH_C2[4] * (xx - yy) * sh_p[8];

			if (deg > 2)
			{
				result_p = result_p +
					SH_C3[0] * y * (3.0f * xx - yy) * sh_p[9] +
					SH_C3[1] * xy * z * sh_p[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_p[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_p[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_p[13] +
					SH_C3[5] * z * (xx - yy) * sh_p[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh_p[15];
			}
		}
	}
	result_p += 0.5f;

	// Remove phase_dc
	result_p.x = result_p.x - 0.5f - SH_C0 * sh_p[0].x;

	// Amplitude is also clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped_p[idx] = (result_p.y < 0);
	if (result_p.y < 0) 
	{
		result_p.y = 0.0f;
	}
	return result_p;
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


// // Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// // given a 2D gaussian parameters.
// __device__ void compute_normal(
// 	const float3& p_orig,
// 	const glm::vec3 scale,
// 	float mod,
// 	const glm::vec4 rot,
// 	const float* projmatrix,
// 	const float* viewmatrix,
// 	const int W,
// 	const int H, 
// 	float3 &normal
// ) {

// 	glm::mat3 R = quat_to_rotmat(rot);
// 	glm::mat3 S = scale_to_mat(scale, mod);
// 	glm::mat3 L = R * S;

// 	// center of Gaussians in the camera coordinate
// 	glm::mat3x4 splat2world = glm::mat3x4(
// 		glm::vec4(L[0], 0.0),
// 		glm::vec4(L[1], 0.0),
// 		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
// 	);

// 	glm::mat4 world2ndc = glm::mat4(
// 		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
// 		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
// 		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
// 		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
// 	);

// 	glm::mat3x4 ndc2pix = glm::mat3x4(
// 		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
// 		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
// 		glm::vec4(0.0, 0.0, 0.0, 1.0)
// 	);

// 	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
// }

// Perform initial steps for each Gaussian prior to rasterization.
template<int C, int PH, int CW>
__global__ void preprocessCUDA(int P, int D, int M, int M_p,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths, float* dists_to_light_ndc,
	float* cov3Ds,
	float* rgb, float* real_img_amp, 
	float* dists_to_light, 
	float* phase_amplitude_from_sh, 
	float4* conic_opacity, float3* normal,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float near_n, float far_n, float dist2phase, bool use_view_dependent_phase, float phase_offset, float dc_offset)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view, near_n, far_n))
		return;

	// float3 nor;
	// compute_normal(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, nor);
	// float cos = -sumf3(p_view * nor);
	// float multiplier = cos > 0 ? 1: -1;
	// nor = multiplier * nor;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if ((colors_precomp != nullptr))
	{
		rgb[idx * C + 0] = colors_precomp[idx * C + 0];
		rgb[idx * C + 1] = colors_precomp[idx * C + 1];
		rgb[idx * C + 2] = colors_precomp[idx * C + 2];
	}

	if ((shs != nullptr))
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	float dist_to_light = sqrtf(p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z);
	float dist_to_light_ndc = far_n / (far_n - near_n) * (1 - near_n / dist_to_light);
	float factor = 1.0f / (dist_to_light * dist_to_light); // falloff

	if ((phasors_precomp != nullptr))
	{
		float phase = dist_to_light * dist2phase;
		float phase_sh = phasors_precomp[idx * CW + 0];
		float amplitude = phasors_precomp[idx * CW + 1];

		phase_amplitude_from_sh[idx * CW + 0] = phase_sh;
		phase_amplitude_from_sh[idx * CW + 1] = amplitude;

		if (use_view_dependent_phase)
		{
			phase += phase_sh;
		}

		real_img_amp[idx * PH + 0] = cosf(phase) * amplitude * factor;
		real_img_amp[idx * PH + 1] = sinf(phase) * amplitude * factor;
		real_img_amp[idx * PH + 2] = amplitude * factor;
		// Quad
		real_img_amp[idx * PH + 3] = (cosf(phase) + dc_offset) * amplitude * factor;
		real_img_amp[idx * PH + 4] = (-cosf(phase) + dc_offset) * amplitude * factor;
		real_img_amp[idx * PH + 5] = (sinf(phase) + dc_offset) * amplitude * factor;
		real_img_amp[idx * PH + 6] = (-sinf(phase) + dc_offset) * amplitude * factor;
	} 

	if ((shs_p != nullptr)) 
	{
		glm::vec2 result_p = computePhasorFromSH(idx, D, M_p, (glm::vec3*)orig_points, *cam_pos, shs_p, clamped_p);
		float phase = dist_to_light * dist2phase + phase_offset;

		phase_amplitude_from_sh[idx * CW + 0] = result_p.x;
		phase_amplitude_from_sh[idx * CW + 1] = result_p.y;

		if (use_view_dependent_phase) phase += result_p.x;

		real_img_amp[idx * PH + 0] = cosf(phase) * result_p.y * factor;
		real_img_amp[idx * PH + 1] = sinf(phase) * result_p.y * factor;
		real_img_amp[idx * PH + 2] = result_p.y * factor;
		// Quad
		real_img_amp[idx * PH + 3] = (cosf(phase) + dc_offset) * result_p.y * factor;
		real_img_amp[idx * PH + 4] = (-cosf(phase) + dc_offset) * result_p.y * factor;
		real_img_amp[idx * PH + 5] = (sinf(phase) + dc_offset) * result_p.y * factor;
		real_img_amp[idx * PH + 6] = (-sinf(phase) + dc_offset) * result_p.y * factor;
	}

	// Store some useful helper data for the next steps.
	dists_to_light[idx] = dist_to_light;
	depths[idx] = p_view.z;
	dists_to_light_ndc[idx] = dist_to_light_ndc;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	// normal[idx] = {nor.x, nor.y, nor.z};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, uint32_t CHANNELS_PHASOR, uint32_t CHANNELS_PHASE, uint32_t SAMPLES_EACH_RAY>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features, const float* __restrict__ features_p,
	const float* __restrict__ features_dists_to_light,
	const float4* __restrict__ conic_opacity, const float3* __restrict__ normal, const float* __restrict__ dists_to_light_ndc,
	float* __restrict__ final_T,
	float* __restrict__ final_alpha_total,
	float* __restrict__ final_DD_D, float* __restrict__ final_DD_D2,
	float* __restrict__ final_AD_D, float* __restrict__ final_AD_D2,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color, float* __restrict__ out_phasor, // out_phasor = real im amp tofType0-3
	float* __restrict__ out_depth, float* __restrict__ out_normal, float* __restrict__ out_acc, 
	float* __restrict__ out_entropy, float* __restrict__ out_depth_distortion, float* __restrict__ out_amp_distortion, float* __restrict__ pixels, 
	float* __restrict__ out_distribution, 
	bool debug
	) 
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	// __shared__ float3 collected_normal[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float P[CHANNELS_PHASOR] = { 0 };
	float D = 0;
	float A = 0;
	// float alpha_total = 0;
	// float alpha_entropy = 0;
	float DD = 0;
	float DD_D = 0;
	float DD_D2 = 0;
	// float AD = 0;
	// float AD_D = 0;
	// float AD_D2 = 0;
	// float N[3] = {0};

	// D[1] = 15.0;

	int isFirstOne = 1;
	float WD[SAMPLES_EACH_RAY * 3] = { 0 };
	int gs_idx = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			// collected_normal[block.thread_rank()] = normal[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
			float w_p = alpha * T * T;

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) 
			{
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			}

			for (int ch = 0; ch < CHANNELS_PHASOR; ch++) 
			{
				P[ch] += features_p[collected_id[j] * CHANNELS_PHASOR + ch] * w_p;
			}

			D += features_dists_to_light[collected_id[j]] * w;

			if (gs_idx < SAMPLES_EACH_RAY)
			{
				WD[gs_idx] = alpha;
				WD[gs_idx+SAMPLES_EACH_RAY] = features_dists_to_light[collected_id[j]];
				WD[gs_idx+SAMPLES_EACH_RAY*2] = features_p[collected_id[j] * CHANNELS_PHASOR + 2];
			}
			gs_idx += 1;

			// if (T > 0.5f && test_T < 0.5)
			// 	D[1] = features_dists_to_light[collected_id[j]];

			float z = dists_to_light_ndc[collected_id[j]];
			// DD += alpha * (z * z * alpha_total - 2.0f * z * DD_D + DD_D2);
			// DD_D += alpha * z;
			// DD_D2 += alpha * z * z;
			DD += w * (z * z * A - 2.0f * z * DD_D + DD_D2);
			DD_D += w * z;
			DD_D2 += w * z * z;

			// alpha_entropy += -alpha * log(alpha);
			// alpha_total += alpha;

			// float amplitude = features_p[collected_id[j] * CHANNELS_PHASOR + 2] * features_dists_to_light[collected_id[j]] * features_dists_to_light[collected_id[j]];
			// // AD += w * (amplitude * amplitude * A - 2.0f * amplitude * AD_D + AD_D2);
			// // AD_D += w * amplitude;
			// // AD_D2 += w * amplitude * amplitude;
			// AD += w * (T * T * amplitude * amplitude * A - 2.0f * amplitude * T * AD_D + AD_D2);
			// AD_D += w * amplitude * T;
			// AD_D2 += w * amplitude * amplitude * T * T;

			// float nor[3] = {collected_normal[j].x, collected_normal[j].y, collected_normal[j].z};
			// for (int ch=0; ch<3; ch++) N[ch] += nor[ch] * w;

			if (pix_id == 38500 && debug)
			{
				float amp = features_p[collected_id[j] * CHANNELS_PHASOR + 2] * features_dists_to_light[collected_id[j]] * features_dists_to_light[collected_id[j]];
				float real;
				float imag;
				if (amp != 0.0f) 
				{
					real = features_p[collected_id[j] * CHANNELS_PHASOR + 0] / amp;
					imag = features_p[collected_id[j] * CHANNELS_PHASOR + 1] / amp;
				}
				else 
				{
					real = 0.0f;
					imag = 0.0f;
				}
				if (isFirstOne) 
				{
					printf("pix_id: %d\nalpha: %f; T: %f; dists_to_light: %f; dist_to_light_ndc: %f; amp: %f; real: %f; imag: %f; amp*T: %f; amp*T/dist2: %f.\n", pix_id, alpha, T, features_dists_to_light[collected_id[j]], dists_to_light_ndc[collected_id[j]], amp, real, imag, amp*T, features_p[collected_id[j] * CHANNELS_PHASOR + 2]*T);
					// printf("pix_id: %d\nalpha: %f; T: %f; dists_to_light: %f; dist_to_light_ndc: %f; amp: %f; amp/dist2: %f.\n", pix_id, alpha, T, features_dists_to_light[collected_id[j]], dists_to_light_ndc[collected_id[j]], amp, features_p[collected_id[j] * CHANNELS_PHASOR + 2]);
					isFirstOne = 0;
				}
				else 
				{
					printf("alpha: %f; T: %f; dists_to_light: %f; dist_to_light_ndc: %f; amp: %f; real: %f; imag: %f; amp*T: %f; amp*T/dist2: %f.\n", alpha, T, features_dists_to_light[collected_id[j]], dists_to_light_ndc[collected_id[j]], amp, real, imag, amp*T, features_p[collected_id[j] * CHANNELS_PHASOR + 2]*T);
					// printf("alpha: %f; T: %f; dists_to_light: %f; dist_to_light_ndc: %f; amp: %f; amp/dist2: %f.\n", alpha, T, features_dists_to_light[collected_id[j]], dists_to_light_ndc[collected_id[j]], amp, features_p[collected_id[j] * CHANNELS_PHASOR + 2]);
				}
			}

			A += alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			atomicAdd(&(pixels[collected_id[j]]), 1.0f);
		}
		block.sync();
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		// final_alpha_total[pix_id] = alpha_total;

		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
		{
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch * H * W + pix_id];
		}

		for (int ch = 0; ch < CHANNELS_PHASOR; ch++)
		{
			out_phasor[ch * H * W + pix_id] = P[ch] + T * bg_color[ch * H * W + pix_id];
		}

		out_depth[pix_id] = D;
		out_acc[pix_id] = A;

		// if (alpha_total >= 1.0f / 255.0f)
		// 	out_entropy[pix_id] = alpha_entropy / alpha_total + log(alpha_total);
		// else
		// 	out_entropy[pix_id] = 0.0f;

		final_DD_D[pix_id] = DD_D;
		final_DD_D2[pix_id] = DD_D2;
		out_depth_distortion[pix_id] = DD;
		// final_AD_D[pix_id] = AD_D;
		// final_AD_D2[pix_id] = AD_D2;
		// out_amp_distortion[pix_id] = AD;

		// for (int ch=0; ch<3; ch++) out_normal[ch * H * W + pix_id] = N[ch];

		for (int ch = 0; ch < SAMPLES_EACH_RAY; ch++)
		{
			out_distribution[ch * H * W + pix_id] = WD[ch];
			out_distribution[(ch+SAMPLES_EACH_RAY) * H * W + pix_id] = WD[ch+SAMPLES_EACH_RAY];
			out_distribution[(ch+SAMPLES_EACH_RAY*2) * H * W + pix_id] = WD[ch+SAMPLES_EACH_RAY*2];
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors, const float* real_img_amp,
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
	float* out_entropy, float* out_depth_distortion, float* out_amp_distortion, float* pixels, 
	float* out_distribution, 
	bool debug
	)
{
	renderCUDA<NUM_CHANNELS, NUM_CHANNELS_PHASOR, NUM_CHANNELS_PHASE, NUM_SAMPLES_EACH_RAY> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors, real_img_amp,
		dists_to_light,
		conic_opacity, normal, dists_to_light_ndc,
		final_T,
		final_alpha_total,
		final_DD_D, final_DD_D2,
		final_AD_D, final_AD_D2,
		n_contrib,
		bg_color,
		out_color, out_phasor, 
		out_depth, out_normal, out_acc, 
		out_entropy, out_depth_distortion, out_amp_distortion, 
		pixels, 
		out_distribution, 
		debug
		);
}

void FORWARD::preprocess(int P, int D, int M, int M_p,
	const float* means3D,
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
	float2* means2D,
	float* depths, float* dists_to_light_ndc,
	float* cov3Ds,
	float* rgb, float* real_img_amp, 
	float* dists_to_light, 
	float* phase_amplitude_from_sh, 
	float4* conic_opacity, float3* normal,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float near_n, float far_n, float depth_range, bool use_view_dependent_phase, float phase_offset, float dc_offset)
{
	const float dist2phase = 4.0f * PI / depth_range;
	preprocessCUDA<NUM_CHANNELS, NUM_CHANNELS_PHASOR, NUM_CHANNELS_CWTOF> << <(P + 255) / 256, 256 >> > (
		P, D, M, M_p,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs, shs_p,
		clamped, clamped_p,
		cov3D_precomp,
		colors_precomp, phasors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths, dists_to_light_ndc,
		cov3Ds,
		rgb, real_img_amp, 
		dists_to_light, 
		phase_amplitude_from_sh,
		conic_opacity, normal,
		grid,
		tiles_touched,
		prefiltered,
		near_n, far_n, dist2phase, use_view_dependent_phase, phase_offset, dc_offset
		);
}