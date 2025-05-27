#ifndef PREWITT_FILTER_H
#define PREWITT_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"
// Function declarations for kernel generation
void get_prewitt_horizontal_kernel_2d(float** kernel);

/**
 * @brief Retrieves the 2D vertical Prewitt kernel.
 *
 * This function initializes the 2D vertical Prewitt kernel used for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 2D vertical Prewitt kernel will be stored.
 */
void get_prewitt_vertical_kernel_2d(float** kernel);

/**
 * @brief Retrieves the 3D horizontal Prewitt kernel.
 *
 * This function initializes the 3D horizontal Prewitt kernel used for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 3D horizontal Prewitt kernel will be stored.
 */
void get_prewitt_horizontal_kernel_3d(float** kernel);

/**
 * @brief Retrieves the 3D vertical Prewitt kernel.
 *
 * This function initializes the 3D vertical Prewitt kernel used for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 3D vertical Prewitt kernel will be stored.
 */
void get_prewitt_vertical_kernel_3d(float** kernel);

/**
 * @brief Retrieves the 3D depth Prewitt kernel.
 *
 * This function initializes the 3D depth Prewitt kernel used for edge detection.
 *
 * @param[out] kernel Pointer to the location where the 3D depth Prewitt kernel will be stored.
 */
void get_prewitt_depth_kernel_3d(float** kernel);

// Function declarations for CUDA kernels
template <typename dtype>
__global__ void prewitt_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, int idz, int xsize, int ysize,
                                         int zsize);

template <typename dtype>
__global__ void prewitt_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, float* deviceKernelDepth,
                                         int xsize, int ysize, int depth);

// Function declaration for prewitt filtering
template <typename dtype>
void prewitt_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

//chunked version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void prewittFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                        const int ysize, const int zsize, const int verbose,
                        int padding_bottom, int padding_top,
                        kernel_dtype* kernel,
                        int kernel_xsize, int kernel_ysize, int kernel_zsize);

template<typename in_dtype, typename out_dtype>
void prewittFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                          const int xsize, const int ysize, const int zsize,
                          const int verbose, int ngpus, const float safetyMargin);

#endif  // PREWITT_FILTER_H
