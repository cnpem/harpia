#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief Retrieves the 2D horizontal Sobel kernel.
 *
 * This function returns the 2D horizontal Sobel kernel for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 2D horizontal Sobel kernel will be stored.
 */
void get_sobel_horizontal_kernel_2d(float** kernel);

/**
 * @brief Retrieves the 2D vertical Sobel kernel.
 *
 * This function returns the 2D vertical Sobel kernel for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 2D vertical Sobel kernel will be stored.
 */
void get_sobel_vertical_kernel_2d(float** kernel);

/**
 * @brief Retrieves the 3D horizontal Sobel kernel.
 *
 * This function returns the 3D horizontal Sobel kernel for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 3D horizontal Sobel kernel will be stored.
 */
void get_sobel_horizontal_kernel_3d(float** kernel);

/**
 * @brief Retrieves the 3D vertical Sobel kernel.
 *
 * This function returns the 3D vertical Sobel kernel for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 3D vertical Sobel kernel will be stored.
 */
void get_sobel_vertical_kernel_3d(float** kernel);

/**
 * @brief Retrieves the 3D depth Sobel kernel.
 *
 * This function returns the 3D depth Sobel kernel for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 3D depth Sobel kernel will be stored.
 */
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


//chunked version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void sobelFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                        const int ysize, const int zsize, const int verbose,
                        int padding_bottom, int padding_top,
                        kernel_dtype* kernel,
                        int kernel_xsize, int kernel_ysize, int kernel_zsize);


template<typename in_dtype, typename out_dtype>
void sobelFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                          const int xsize, const int ysize, const int zsize,
                          const int verbose, int ngpus, const float safetyMargin);
#endif  // SOBEL_FILTER_H
