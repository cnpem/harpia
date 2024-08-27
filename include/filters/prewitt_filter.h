#ifndef PREWITT_FILTER_H
#define PREWITT_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"
// Function declarations for kernel generation
void get_prewitt_horizontal_kernel_2d(float** kernel);
void get_prewitt_vertical_kernel_2d(float** kernel);
void get_prewitt_horizontal_kernel_3d(float** kernel);
void get_prewitt_vertical_kernel_3d(float** kernel);
void get_prewitt_depth_kernel_3d(float** kernel);

// Function declarations for CUDA kernels
template <typename dtype>
__global__ void prewitt_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, int idz, int xsize, int ysize,
                                         int zsize);

template <typename dtype>
__global__ void prewitt_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, float* deviceKernelDepth,
                                         int xsize, int ysize, int depth);

// Function declaration for prewitt filtering
template <typename dtype>
void prewitt_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

#endif  // PREWITT_FILTER_H
