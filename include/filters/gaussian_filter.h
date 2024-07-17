#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include"../common/convolution.h"
#include"../common/kernels.h"


// CUDA kernel for 2D Gaussian filtering
template<typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, float* dev_kernel,
                                          int idz, int rows, int cols, int slices, int rows_kernel, int cols_kernel);

// CUDA kernel for 3D Gaussian filtering
template<typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, float* dev_kernel,
                                          int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

// Host function for Gaussian filtering
template<typename dtype>
void gaussian_filtering(dtype* image, float* output, int rows, int cols, int slices, float sigma, bool type);


#endif // GAUSSIAN_FILTER_H
