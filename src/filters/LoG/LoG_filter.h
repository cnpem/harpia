#ifndef LOG_FILTER_H
#define LOG_FILTER_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Function to generate a 2D Laplacian kernel
void get_laplacian_kernel_2d(float** kernel);

// Function to generate a 3D Laplacian kernel
void get_laplacian_kernel_3d(float** kernel);

// CUDA kernel for 2D Laplacian of Gaussian filtering
template<typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* dev_kernel, int idz, int rows, int cols, int slices);

// CUDA kernel for 3D Laplacian of Gaussian filtering
template<typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* dev_kernel, int rows, int cols, int depth);

// Host function for Laplacian of Gaussian filtering
template<typename dtype>
void log_filtering(dtype* image, float* output, int rows, int cols, int slices, bool type);

#endif // LOG_FILTER_H
