#ifndef MEAN_FILTER_H
#define MEAN_FILTER_H

#include <iostream>
#include <cuda_runtime.h>
#include"../common/kernels.h"
#include"../common/convolution.h"

// Kernel to compute the mean for a 2D region
template<typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int xsize, int cols, int kx, int ky);

// Kernel to compute the mean for a 3D region
template<typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean, int i, int j, int k, int xsize, int ysize, int zsize, int kx, int ky, int kz);

// 2D mean filter kernel
template<typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int xsize, int ysize, int idz, int kx, int ky);

// 3D mean filter kernel
template<typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);

// Host function to call the mean filtering
template<typename dtype>
void mean_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);

#endif // MEAN_FILTER_H