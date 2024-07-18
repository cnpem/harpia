
#ifndef SAUVOLA_H
#define SAUVOLA_H

#include<iostream>
#include<cuda_runtime.h>
#include"../common/kernels.h"

// 2D Sauvola Kernel
template<typename dtype>
__global__ void sauvola_kernel_2d(dtype* image, float* output, float weight, dtype range, int rows, int cols, int idz, int rows_kernel, int cols_kernel);

// 3D Sauvola Kernel
template<typename dtype>
__global__ void sauvola_kernel_3d(dtype* image, float* output, float weight, dtype range, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

// Sauvola Threshold Function
template<typename dtype>
void sauvola_threshold(dtype* image, float* output, float weight, dtype range, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);

#endif // SAUVOLA_H