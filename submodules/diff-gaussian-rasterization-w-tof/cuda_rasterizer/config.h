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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB

#define NUM_CHANNELS_CWTOF 2 // (continuous wave ToF) PHASE, AMPLITUDE
#define NUM_CHANNELS_PHASOR 7 // REAL, IMAGINARY, AMPLITUDE, tofQuad0, tofQuad1, tofQuad2, tofQuad3 
#define NUM_CHANNELS_PHASE 2 // PHASE -> REAL, IMAGINARY from SH
#define NUM_SAMPLES_EACH_RAY 1 // number of samples to return along each ray

#define BLOCK_X 16
#define BLOCK_Y 16

#endif