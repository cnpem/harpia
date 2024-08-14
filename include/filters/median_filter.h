#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include <cuda_runtime.h>
#include <chrono>

// Forward declarations of device functions
template<typename dtype>
__device__ void bubble_sort(dtype *array, int size);

template<typename dtype>
__device__ void get_median_kernel_2d(dtype* image, dtype* kernel, int i, int j, int xsize, int ysize, int kx, int ky);

template<typename dtype>
__device__ void get_median_kernel_3d(dtype* image, dtype* kernel,
                          int i, int j, int k, 
                          int xsize, int ysize, int zsize,
                          int kx, int ky, int kz);

// Kernel function declarations
template<typename dtype>
__global__ void median_filter_kernel_2d(dtype* image, dtype* output, dtype* kernel, int xsize, int ysize, int idz, int kx, int ky);

template<typename dtype>
__global__ void median_filter_kernel_3d(dtype* image, dtype* output, dtype* kernel,
                                        int xsize, int ysize, int zsize, int idz,
                                        int kx, int ky, int kz);

// Function to call from host
template<typename dtype>
void median_filtering(dtype* image, dtype* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);


#endif // MEDIAN_FILTER_H
