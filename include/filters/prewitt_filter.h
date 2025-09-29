#ifndef PREWITT_FILTER_H
#define PREWITT_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief Initializes the 2D horizontal Prewitt kernel.
 *
 * @param[out] kernel Pointer to the location where the horizontal kernel will be stored.
 */
void get_prewitt_horizontal_kernel_2d(float** kernel);

/**
 * @brief Initializes the 2D vertical Prewitt kernel.
 *
 * @param[out] kernel Pointer to the location where the vertical kernel will be stored.
 */
void get_prewitt_vertical_kernel_2d(float** kernel);

/**
 * @brief Initializes the 3D horizontal Prewitt kernel.
 *
 * @param[out] kernel Pointer to the location where the horizontal kernel will be stored.
 */
void get_prewitt_horizontal_kernel_3d(float** kernel);

/**
 * @brief Initializes the 3D vertical Prewitt kernel.
 *
 * @param[out] kernel Pointer to the location where the vertical kernel will be stored.
 */
void get_prewitt_vertical_kernel_3d(float** kernel);

/**
 * @brief Initializes the 3D depth Prewitt kernel.
 *
 * @param[out] kernel Pointer to the location where the depth kernel will be stored.
 */
void get_prewitt_depth_kernel_3d(float** kernel);

/**
 * @brief CUDA kernel for applying 2D Prewitt filtering on a single slice.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] image Pointer to input image in device memory.
 * @param[out] output Pointer to output image in device memory.
 * @param[in] deviceKernelHorizontal Pointer to horizontal kernel in device memory.
 * @param[in] deviceKernelVertical Pointer to vertical kernel in device memory.
 * @param[in] idz Slice index (along Z).
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image volume.
 */
template <typename dtype>
__global__ void prewitt_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, int idz, int xsize, int ysize,
                                         int zsize);

/**
 * @brief CUDA kernel for applying 3D Prewitt filtering.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] image Pointer to input image in device memory.
 * @param[out] output Pointer to output image in device memory.
 * @param[in] deviceKernelHorizontal Pointer to horizontal kernel in device memory.
 * @param[in] deviceKernelVertical Pointer to vertical kernel in device memory.
 * @param[in] deviceKernelDepth Pointer to depth kernel in device memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] depth Depth of the image.
 */
template <typename dtype>
__global__ void prewitt_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, float* deviceKernelDepth,
                                         int xsize, int ysize, int depth);

/**
 * @brief Host function to apply the Prewitt filter on a 2D or 3D image.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] image Pointer to input image.
 * @param[out] output Pointer to output image.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type If true, use 3D filter; otherwise, apply 2D slice-wise filtering.
 */
template <typename dtype>
void prewitt_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

/**
 * @brief Chunked execution of 3D Prewitt filter with kernel reuse across tiles.
 *
 * @tparam in_dtype Input data type.
 * @tparam out_dtype Output data type.
 * @tparam kernel_dtype Kernel data type.
 * @param[in] hostImage Pointer to host input image.
 * @param[out] hostOutput Pointer to host output image.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] verbose Verbose flag for debugging.
 * @param[in] padding_bottom Amount of padding at the bottom along the z-dimension.
 * @param[in] padding_top Amount of padding at the top along the z-dimension.
 * @param[in] kernel Pointer to the convolution kernel (flattened).
 * @param[in] kernel_xsize Kernel width.
 * @param[in] kernel_ysize Kernel height.
 * @param[in] kernel_zsize Kernel depth.
 */
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void prewittFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                        const int ysize, const int zsize, const int verbose,
                        int padding_bottom, int padding_top,
                        kernel_dtype* kernel,
                        int kernel_xsize, int kernel_ysize, int kernel_zsize);

/**
 * @brief Chunked execution of Prewitt filtering with automatic GPU selection and memory constraints.
 *
 * @tparam in_dtype Input data type.
 * @tparam out_dtype Output data type.
 * @param[in] hostImage Pointer to host input image.
 * @param[out] hostOutput Pointer to host output image.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type3d Type flag (true for 3D filtering, false for 2D slice-wise).
 * @param[in] verbose Verbose output toggle.
 * @param[in] ngpus Number of GPUs to use.
 * @param[in] safetyMargin Fraction of memory to leave free on each GPU.
 */
template<typename in_dtype, typename out_dtype>
void prewittFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                          const int xsize, const int ysize, const int zsize, const int type3d,
                          const int verbose, int ngpus, const float safetyMargin);

#endif  // PREWITT_FILTER_H