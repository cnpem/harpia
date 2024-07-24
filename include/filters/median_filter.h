#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include <cuda_runtime.h>
#include <chrono>

// Forward declarations of device functions
template<typename dtype>
__device__ void bubble_sort(dtype *array, int size);

template<typename dtype>
__device__ void get_median_kernel_2d(dtype* image, dtype* kernel, int i, int j, int rows, int cols, int rows_kernel, int cols_kernel);

template<typename dtype>
__device__ void get_median_kernel_3d(dtype* image, dtype* kernel,
                          int i, int j, int k, 
                          int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel);

// Kernel function declarations
template<typename dtype>
__global__ void median_filter_kernel_2d(dtype* image, dtype* output, dtype* kernel, int rows, int cols, int idz, int rows_kernel, int cols_kernel);

template<typename dtype>
__global__ void median_filter_kernel_3d(dtype* image, dtype* output, dtype* kernel,
                                        int rows, int cols, int depth, int idz,
                                        int rows_kernel, int cols_kernel, int depth_kernel);

// Function to call from host
template<typename dtype>
void median_filtering(dtype* image, dtype* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);


#endif // MEDIAN_FILTER_H
