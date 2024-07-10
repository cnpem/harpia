#ifndef MEAN_FILTER_H
#define MEAN_FILTER_H

#include <iostream>
#include <cuda_runtime.h>


// Kernel to compute the mean for a 2D region
template<typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int rows, int cols, int rows_kernel, int cols_kernel);

// Kernel to compute the mean for a 3D region
template<typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean, int i, int j, int k, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

// 2D mean filter kernel
template<typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int rows, int cols, int idz, int rows_kernel, int cols_kernel);

// 3D mean filter kernel
template<typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

// Host function to call the mean filtering
template<typename dtype>
void mean_filtering(dtype* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

#endif // MEAN_FILTER_H