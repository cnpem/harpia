#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

// CUDA kernel for 2D Gaussian filtering
template <typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz,
                                          int xsize, int ysize, int zsize, int nx, int ny);

// CUDA kernel for 3D Gaussian filtering
template <typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, float* deviceKernel,
                                          int xsize, int ysize, int zsize, int nx, int ny, int nz);

// Host function for Gaussian filtering
template <typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma,
                        bool type);



//chunked executor version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void gaussianFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize);
                    
template<typename in_dtype, typename out_dtype>
void gaussianFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize, float sigma,
                      const int verbose, int ngpus,const float safetyMargin);

#endif  // GAUSSIAN_FILTER_H
