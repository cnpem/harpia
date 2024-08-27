#ifndef MEAN_FILTER_H
#define MEAN_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

// Kernel to compute the mean for a 2D region
template <typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int xsize, int cols,
                                   int nx, int ny);

// Kernel to compute the mean for a 3D region
template <typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean, int i, int j, int k, int xsize,
                                   int ysize, int zsize, int nx, int ny, int nz);

// 2D mean filter kernel
template <typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int xsize, int ysize, int idz,
                                      int nx, int ny);

// 3D mean filter kernel
template <typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output, int xsize, int ysize, int zsize,
                                      int nx, int ny, int nz);

// Host function to call the mean filtering
template <typename dtype>
void mean_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, int nx, int ny,
                    int nz);

#endif  // MEAN_FILTER_H