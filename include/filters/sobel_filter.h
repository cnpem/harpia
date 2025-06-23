#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief Initializes the 2D horizontal Sobel kernel.
 *
 * @param[out] kernel Pointer to the memory where the horizontal kernel will be stored.
 */
void get_sobel_horizontal_kernel_2d(float** kernel);

/**
 * @brief Initializes the 2D vertical Sobel kernel.
 *
 * @param[out] kernel Pointer to the memory where the vertical kernel will be stored.
 */
void get_sobel_vertical_kernel_2d(float** kernel);

/**
 * @brief Initializes the 3D horizontal Sobel kernel.
 *
 * @param[out] kernel Pointer to the memory where the horizontal kernel will be stored.
 */
void get_sobel_horizontal_kernel_3d(float** kernel);

/**
 * @brief Initializes the 3D vertical Sobel kernel.
 *
 * @param[out] kernel Pointer to the memory where the vertical kernel will be stored.
 */
void get_sobel_vertical_kernel_3d(float** kernel);

/**
 * @brief Initializes the 3D depth Sobel kernel.
 *
 * @param[out] kernel Pointer to the memory where the depth kernel will be stored.
 */
void get_sobel_depth_kernel_3d(float** kernel);

/**
 * @brief CUDA kernel for 2D Sobel filtering applied on a specific slice (Z-plane).
 *
 * @tparam dtype Type of the input image (e.g., unsigned char, float).
 * @param[in] image Pointer to the input image in device memory.
 * @param[out] output Pointer to the output image in device memory.
 * @param[in] deviceKernelHorizontal Horizontal kernel (in device memory).
 * @param[in] deviceKernelVertical Vertical kernel (in device memory).
 * @param[in] idz Index of the current Z-slice.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image volume.
 */
template <typename dtype>
__global__ void sobel_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
                                       float* deviceKernelVertical, int idz, int xsize, int ysize,
                                       int zsize);

/**
 * @brief CUDA kernel for 3D Sobel filtering.
 *
 * @tparam dtype Type of the input image.
 * @param[in] image Pointer to the input image in device memory.
 * @param[out] output Pointer to the output image in device memory.
 * @param[in] deviceKernelHorizontal Horizontal kernel (in device memory).
 * @param[in] deviceKernelVertical Vertical kernel (in device memory).
 * @param[in] deviceKernelDepth Depth kernel (in device memory).
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image volume.
 */
template <typename dtype>
__global__ void sobel_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                       float* deviceKernelVertical, float* deviceKernelDepth,
                                       int xsize, int ysize, int zsize);

/**
 * @brief Host function to apply the Sobel filter on 2D or 3D images.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] image Pointer to the input image.
 * @param[out] output Pointer to the output image.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type If true, applies 3D filtering; if false, applies 2D filtering per slice.
 */
template <typename dtype>
void sobel_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

/**
 * @brief Applies 3D Sobel filtering with kernel reuse, optimized for large volumes using chunking.
 *
 * @tparam in_dtype Type of the input image.
 * @tparam out_dtype Type of the output image.
 * @tparam kernel_dtype Type of the kernel values.
 * @param[in] hostImage Pointer to the host image data.
 * @param[out] hostOutput Pointer to the host output buffer.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] verbose Verbosity level.
 * @param[in] padding_bottom Padding size at the bottom (used in chunked processing).
 * @param[in] padding_top Padding size at the top (used in chunked processing).
 * @param[in] kernel Pointer to the kernel data (flattened).
 * @param[in] kernel_xsize Width of the kernel.
 * @param[in] kernel_ysize Height of the kernel.
 * @param[in] kernel_zsize Depth of the kernel.
 */
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void sobelFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                      const int ysize, const int zsize, const int verbose,
                      int padding_bottom, int padding_top,
                      kernel_dtype* kernel,
                      int kernel_xsize, int kernel_ysize, int kernel_zsize);

/**
 * @brief Chunked Sobel filter execution across slices or volumes, optimized for GPU memory.
 *
 * @tparam in_dtype Type of the input image.
 * @tparam out_dtype Type of the output image.
 * @param[in] hostImage Pointer to the input image on host.
 * @param[out] hostOutput Pointer to the output image on host.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type3d If true, applies 3D filtering; otherwise, 2D filtering per slice.
 * @param[in] verbose Verbosity level.
 * @param[in] ngpus Number of GPUs to use.
 * @param[in] safetyMargin Fraction of memory to reserve on each GPU.
 */
template<typename in_dtype, typename out_dtype>
void sobelFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                        const int xsize, const int ysize, const int zsize, const int type3d,
                        const int verbose, int ngpus, const float safetyMargin);

#endif  // SOBEL_FILTER_H