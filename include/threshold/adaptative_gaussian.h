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


template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void adaptativeGaussianThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize, float weight);

template<typename in_dtype, typename out_dtype>
void adaptativeGaussianThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize, float sigma, float weight, const int type3d,
                      const int verbose, int ngpus,const float safetyMargin );

#endif  // ADAPTATIVE_GAUSSIAN_H