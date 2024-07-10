#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Function declarations for 2D Sobel kernels
void get_sobel_horizontal_kernel_2d(float** kernel);
void get_sobel_vertical_kernel_2d(float** kernel);

// Function declarations for 3D Sobel kernels
void get_sobel_horizontal_kernel_3d(float** kernel);
void get_sobel_vertical_kernel_3d(float** kernel);
void get_sobel_depth_kernel_3d(float** kernel);

// CUDA kernel for 2D Sobel filtering
template<typename dtype>
__global__ void sobel_filter_kernel_2d(dtype* image, float* output,
                                       float* dev_kernel_horizontal, float* dev_kernel_vertical,
                                       int idz, int rows, int cols, int slices);

// CUDA kernel for 3D Sobel filtering
template<typename dtype>
__global__ void sobel_filter_kernel_3d(dtype* image, float* output,
                                       float* dev_kernel_horizontal, float* dev_kernel_vertical, float* dev_kernel_depth,
                                       int rows, int cols, int depth);

// Template function for launching Sobel filtering on CUDA device memory
template<typename dtype>
void sobel_filtering(dtype* image, float* output, int rows, int cols, int slices, bool type);

#endif // SOBEL_FILTER_H
