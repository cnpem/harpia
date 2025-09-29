#ifndef ADAPTATIVE_GAUSSIAN_H
#define ADAPTATIVE_GAUSSIAN_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief CUDA kernel for 2D adaptive Gaussian thresholding.
 *
 * Applies a local Gaussian filter and compares each pixel to a weighted version of the local mean.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Input image on device memory.
 * @param[out] output Output binary image on device memory.
 * @param[in] dev_kernel Gaussian kernel on device.
 * @param[in] weight Weight to subtract from the local mean (threshold = mean - weight).
 * @param[in] idz Index of the slice being processed.
 * @param[in] rows Number of rows in the image.
 * @param[in] cols Number of columns in the image.
 * @param[in] slices Number of slices (for context, not used directly in 2D).
 * @param[in] rows_kernel Height of the Gaussian kernel.
 * @param[in] cols_kernel Width of the Gaussian kernel.
 */
template <typename dtype>
__global__ void local_gaussian_kernel_2d(dtype* image, float* output, float* dev_kernel,
                                         float weight, int idz, int rows, int cols, int slices,
                                         int rows_kernel, int cols_kernel);

/**
 * @brief CUDA kernel for 3D adaptive Gaussian thresholding.
 *
 * Applies a local 3D Gaussian filter and computes thresholding using a weighted mean.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Input 3D image (flattened) on device memory.
 * @param[out] output Output 3D binary image (flattened).
 * @param[in] dev_kernel 3D Gaussian kernel on device.
 * @param[in] weight Thresholding weight.
 * @param[in] rows Image height.
 * @param[in] cols Image width.
 * @param[in] depth Image depth (z-dimension).
 * @param[in] rows_kernel Kernel height.
 * @param[in] cols_kernel Kernel width.
 * @param[in] depth_kernel Kernel depth.
 */
template <typename dtype>
__global__ void local_gaussian_kernel_3d(dtype* image, float* output, float* dev_kernel,
                                         float weight, int rows, int cols, int depth,
                                         int rows_kernel, int cols_kernel, int depth_kernel);

/**
 * @brief Host function to launch local Gaussian thresholding on 2D or 3D data.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Host-side input image (flattened).
 * @param[out] output Host-side output binary image (flattened).
 * @param[in] rows Image height.
 * @param[in] cols Image width.
 * @param[in] depth Image depth.
 * @param[in] sigma Standard deviation for Gaussian kernel.
 * @param[in] weight Subtracted constant from local mean.
 * @param[in] type If true, apply 3D version; otherwise, apply 2D slice-by-slice.
 */
template <typename dtype>
void local_gaussian_threshold(dtype* image, float* output, int rows, int cols, int depth,
                              float sigma, float weight, bool type);

/**
 * @brief Chunked GPU processing for large 3D adaptive Gaussian thresholding.
 *
 * Suitable for cases where the full volume does not fit in GPU memory.
 *
 * @tparam in_dtype Input data type.
 * @tparam out_dtype Output data type.
 * @tparam kernel_dtype Kernel data type.
 * @param[in] hostImage Input 3D image (host memory).
 * @param[out] hostOutput Output 3D image (host memory).
 * @param[in] xsize Width.
 * @param[in] ysize Height.
 * @param[in] zsize Depth.
 * @param[in] verbose Verbosity level.
 * @param[in] padding_bottom Padding below each chunk.
 * @param[in] padding_top Padding above each chunk.
 * @param[in] kernel Precomputed Gaussian kernel (host memory).
 * @param[in] kernel_xsize Kernel width.
 * @param[in] kernel_ysize Kernel height.
 * @param[in] kernel_zsize Kernel depth.
 * @param[in] weight Subtracted value from local mean.
 */
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void adaptativeGaussianThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                                      const int ysize, const int zsize, const int verbose,
                                      int padding_bottom, int padding_top,
                                      kernel_dtype* kernel,
                                      int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                      float weight);

/**
 * @brief Chunked adaptive Gaussian thresholding (automatic kernel generation).
 *
 * Computes the Gaussian kernel internally based on sigma and launches chunked GPU execution.
 *
 * @tparam in_dtype Input data type.
 * @tparam out_dtype Output data type.
 * @param[in] hostImage Input 3D image (host memory).
 * @param[out] hostOutput Output 3D image (host memory).
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] sigma Standard deviation of the Gaussian kernel.
 * @param[in] weight Subtracted value from local mean.
 * @param[in] type3d Use full 3D kernel if true, otherwise 2D per slice.
 * @param[in] verbose Verbosity flag.
 * @param[in] ngpus Number of GPUs to use.
 * @param[in] safetyMargin Fraction of memory to leave free per GPU.
 */
template<typename in_dtype, typename out_dtype>
void adaptativeGaussianThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput,
                                        const int xsize, const int ysize, const int zsize,
                                        float sigma, float weight, const int type3d,
                                        const int verbose, int ngpus, const float safetyMargin);

#endif  // ADAPTATIVE_GAUSSIAN_H