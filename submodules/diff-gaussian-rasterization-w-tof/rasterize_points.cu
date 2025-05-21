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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    const torch::Tensor& colors, const torch::Tensor& phasors,
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
	const torch::Tensor& sh, const torch::Tensor& sh_p,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const float near_n,
	const float far_n,
	const float depth_range,
	const bool use_view_dependent_phase,
	const float phase_offset, const float dc_offset
	)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_phasor = torch::full({NUM_CHANNELS_PHASOR, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor pixels = torch::zeros({P, 1}, means3D.options());

  // Additional images to render
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor out_acc = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_entropy = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_depth_distortion = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_amp_distortion = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_distribution = torch::full({NUM_SAMPLES_EACH_RAY * 3, H, W}, 0.0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if (sh.defined()) 
	  {
		if(sh.size(0) != 0)
		{
			M = sh.size(1); // max no of sh coefficient --> 1, 4, 9, 16
		}
	  }

	  int M_p = 0;
	  if (sh_p.defined()) 
	  {
		if(sh_p.size(0) != 0)
		{
			M_p = sh_p.size(1);
		}
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M, M_p,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(), sh_p.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), phasors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(), out_phasor.contiguous().data<float>(), 
		out_depth.contiguous().data<float>(), out_normal.contiguous().data<float>(), out_acc.contiguous().data<float>(), 
		out_entropy.contiguous().data<float>(), out_depth_distortion.contiguous().data<float>(), out_amp_distortion.contiguous().data<float>(),
		pixels.contiguous().data<float>(),
		out_distribution.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug, 
		near_n, far_n, depth_range, 
		use_view_dependent_phase,
		phase_offset, dc_offset
		);
  }
  return std::make_tuple(
	rendered, 
	out_color, out_phasor, 
	out_depth, out_normal, out_acc, 
	out_entropy, out_depth_distortion, out_amp_distortion, 
	pixels, 
	out_distribution,
	radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const float phase_offset,
	const float dc_offset
	) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if (sh.defined())
  {
	if(sh.size(0) != 0)
	{	
		M = sh.size(1);
	}
  }

  int M_p = 0;
  if (sh_p.defined())
  {
	if(sh_p.size(0) != 0)
	{	
		M_p = sh_p.size(1);
	}
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dphasors = torch::zeros({P, NUM_CHANNELS_PHASOR}, means3D.options());
  torch::Tensor dL_ddists_to_light = torch::zeros({P, 1}, means3D.options()); // For depth loss. View space
  torch::Tensor dL_dzs_ndc = torch::zeros({P, 1}, means3D.options()); // For depth distortion loss.
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dsh_p = torch::zeros({P, M_p, NUM_CHANNELS_CWTOF}, means3D.options()); // phase & amp
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dphase_offset = torch::zeros({1}, means3D.options());
  torch::Tensor dL_ddc_offset = torch::zeros({1}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, M_p, R,
		background.contiguous().data<float>(),
		W, H, 
		means3D.contiguous().data<float>(),
		sh.contiguous().data<float>(), sh_p.contiguous().data<float>(),
		colors.contiguous().data<float>(), phasors.contiguous().data<float>(),
		scales.data_ptr<float>(),
		scale_modifier,
		rotations.data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		radii.contiguous().data<int>(),
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		dL_dout_color.contiguous().data<float>(), dL_dout_phasor.contiguous().data<float>(), 
		dL_dout_depth.contiguous().data<float>(), dL_dout_normal.contiguous().data<float>(), dL_dout_acc.contiguous().data<float>(),
		dL_dout_entropy.contiguous().data<float>(), dL_dout_depth_distortion.contiguous().data<float>(), dL_dout_amp_distortion.contiguous().data<float>(),
		dL_dmeans2D.contiguous().data<float>(),
		dL_dconic.contiguous().data<float>(),  
		dL_dopacity.contiguous().data<float>(),
		dL_dcolors.contiguous().data<float>(), dL_dphasors.contiguous().data<float>(), dL_ddists_to_light.contiguous().data<float>(), dL_dzs_ndc.contiguous().data<float>(),
		dL_dmeans3D.contiguous().data<float>(),
		dL_dcov3D.contiguous().data<float>(),
		dL_dsh.contiguous().data<float>(), dL_dsh_p.contiguous().data<float>(),
		dL_dscales.contiguous().data<float>(),
		dL_drotations.contiguous().data<float>(),
		dL_dphase_offset.contiguous().data<float>(),
		dL_ddc_offset.contiguous().data<float>(),
		debug, 
		near_n, far_n, depth_range, 
		use_view_dependent_phase,
		phase_offset, dc_offset
		);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dphasors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dsh_p, dL_dscales, dL_drotations, dL_dphase_offset, dL_ddc_offset);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix,
		float znear, float zfar)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>(),
		znear, zfar);
  }
  
  return present;
}