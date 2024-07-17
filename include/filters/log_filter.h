#ifndef LOG_FILTER_H
#define LOG_FILTER_H

#include <iostream>
#include <cuda_runtime.h>
#include"../common/convolution.h"
#include"../common/kernels.h"

// Function declarations
template<typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* dev_kernel, int idz, int rows, int cols, int slices);

template<typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* dev_kernel, int rows, int cols, int depth);

template<typename dtype>
void log_filtering(dtype* image, float* output, int rows, int cols, int slices, bool type);


#endif // LOG_FILTER_H
