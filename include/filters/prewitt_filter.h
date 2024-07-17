#ifndef PREWITT_FILTER_H
#define PREWITT_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include"../common/convolution.h"
#include"../common/kernels.h"
// Function declarations for kernel generation
void get_prewitt_horizontal_kernel_2d(float** kernel);
void get_prewitt_vertical_kernel_2d(float** kernel);
void get_prewitt_horizontal_kernel_3d(float** kernel);
void get_prewitt_vertical_kernel_3d(float** kernel);
void get_prewitt_depth_kernel_3d(float** kernel);

// Function declarations for CUDA kernels
template<typename dtype>
__global__ void prewitt_filter_kernel_2d(dtype* image, float* output,
                                         float* dev_kernel_horizontal, float* dev_kernel_vertical,
                                         int idz, int rows, int cols, int slices);

template<typename dtype>
__global__ void prewitt_filter_kernel_3d(dtype* image, float* output,
                                         float* dev_kernel_horizontal, float* dev_kernel_vertical, float* dev_kernel_depth,
                                         int rows, int cols, int depth);

// Function declaration for prewitt filtering
template<typename dtype>
void prewitt_filtering(dtype* image, float* output, int rows, int cols, int slices, bool type);

#endif // PREWITT_FILTER_H
