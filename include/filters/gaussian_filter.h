#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include"../common/convolution.h"
#include"../common/kernels.h"


// CUDA kernel for 2D Gaussian filtering
template<typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, float* dev_kernel,
                                          int idz, int xsize, int ysize, int zsize, int kx, int ky);

// CUDA kernel for 3D Gaussian filtering
template<typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, float* dev_kernel,
                                          int xsize, int ysize, int zsize, int kx, int ky, int kz);

// Host function for Gaussian filtering
template<typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma, bool type);


#endif // GAUSSIAN_FILTER_H
