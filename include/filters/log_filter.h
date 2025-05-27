#ifndef LOG_FILTER_H
#define LOG_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

// Function declarations
template <typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz,
                                     int xsize, int ysize, int zsize);

template <typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* deviceKernel, int xsize,
                                     int ysize, int zsize);

template <typename dtype>
void log_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

//chunked

template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void logFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize);

template<typename in_dtype, typename out_dtype>
void logFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize,
                      const int verbose, int ngpus, const float safetyMargin);


#endif  // LOG_FILTER_H
