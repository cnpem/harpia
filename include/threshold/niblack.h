#ifndef NIBLACK_H
#define NIBLACK_H

#include <cuda_runtime.h>
#include <iostream>
#include"../filters/mean_filter.h"
#include"../common/kernels.h"
// 2D Niblack Kernel
template<typename dtype>
__global__ void niblack_kernel_2d(dtype* image, float* output, float weight, int rows, int cols, int idz, int rows_kernel, int cols_kernel);

// 3D Niblack Kernel
template<typename dtype>
__global__ void niblack_kernel_3d(dtype* image, float* output, float weight, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

// Niblack Threshold Function
template<typename dtype>
void niblack_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

#endif //NIBLACK_H