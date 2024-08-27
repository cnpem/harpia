#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

// Function to get the 2D horizontal Sobel kernel
void get_sobel_horizontal_kernel_2d(float** kernel);

// Function to get the 2D vertical Sobel kernel
void get_sobel_vertical_kernel_2d(float** kernel);

// Function to get the 3D horizontal Sobel kernel
void get_sobel_horizontal_kernel_3d(float** kernel);

// Function to get the 3D vertical Sobel kernel
void get_sobel_vertical_kernel_3d(float** kernel);

// Function to get the 3D depth Sobel kernel
void get_sobel_depth_kernel_3d(float** kernel);

// Sobel filter kernel for 2D images
template <typename dtype>
__global__ void sobel_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
                                       float* deviceKernelVertical, int idz, int xsize, int ysize,
                                       int zsize);

// Sobel filter kernel for 3D images
template <typename dtype>
__global__ void sobel_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                       float* deviceKernelVertical, float* deviceKernelDepth,
                                       int xsize, int ysize, int zsize);

// Function to apply Sobel filter to an image
template <typename dtype>
void sobel_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

#endif  // SOBEL_FILTER_H
