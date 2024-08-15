#ifndef LOG_FILTER_H
#define LOG_FILTER_H

#include <iostream>
#include <cuda_runtime.h>
#include"../common/convolution.h"
#include"../common/kernels.h"

// Function declarations
template<typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz, int xsize, int ysize, int zsize);

template<typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* deviceKernel, int xsize, int ysize, int zsize);

template<typename dtype>
void log_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);


#endif // LOG_FILTER_H
