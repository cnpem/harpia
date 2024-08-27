#ifndef ADAPTATIVE_GAUSSIAN_H
#define ADAPTATIVE_GAUSSIAN_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"
// Kernel for 2D local Gaussian thresholding
template <typename dtype>
__global__ void local_gaussian_kernel_2d(dtype* image, float* output, float* dev_kernel,
                                         float weight, int idz, int rows, int cols, int slices,
                                         int rows_kernel, int cols_kernel);

// Kernel for 3D local Gaussian thresholding
template <typename dtype>
__global__ void local_gaussian_kernel_3d(dtype* image, float* output, float* dev_kernel,
                                         float weight, int rows, int cols, int depth,
                                         int rows_kernel, int cols_kernel, int depth_kernel);

// Host function for local Gaussian thresholding
template <typename dtype>
void local_gaussian_threshold(dtype* image, float* output, int rows, int cols, int depth,
                              float sigma, float weight, bool type);

#endif  // ADAPTATIVE_GAUSSIAN_H