#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

// CUDA kernel for 2D Gaussian filtering
template <typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz,
                                          int xsize, int ysize, int zsize, int nx, int ny);

// CUDA kernel for 3D Gaussian filtering
template <typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, float* deviceKernel,
                                          int xsize, int ysize, int zsize, int nx, int ny, int nz);

// Host function for Gaussian filtering
template <typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma,
                        bool type);

#endif  // GAUSSIAN_FILTER_H
