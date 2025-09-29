#ifndef NIBLACK_H
#define NIBLACK_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"
#include "../filters/mean_filter.h"

/**
 * @brief CUDA kernel for 2D Niblack thresholding.
 *
 * Computes the threshold using local mean and standard deviation over a neighborhood
 * and applies the Niblack formula: `threshold = mean + weight * stddev`.
 *
 * @tparam dtype Input image type.
 * @param[in] image Input image.
 * @param[out] output Thresholded output.
 * @param[in] weight Weight for stddev in the threshold formula.
 * @param[in] rows Number of image rows.
 * @param[in] cols Number of image columns.
 * @param[in] idz Current z-index (slice index).
 * @param[in] rows_kernel Kernel height.
 * @param[in] cols_kernel Kernel width.
 */
template <typename dtype>
__global__ void niblack_kernel_2d(dtype* image, float* output, float weight, int rows, int cols,
                                  int idz, int rows_kernel, int cols_kernel);

/**
 * @brief CUDA kernel for 3D Niblack thresholding.
 *
 * Applies the Niblack thresholding formula on a 3D neighborhood.
 *
 * @tparam dtype Input image type.
 * @param[in] image Input image volume.
 * @param[out] output Thresholded output.
 * @param[in] weight Weight for stddev in the threshold formula.
 * @param[in] rows Image height.
 * @param[in] cols Image width.
 * @param[in] depth Image depth (number of slices).
 * @param[in] rows_kernel Kernel size in y-dimension.
 * @param[in] cols_kernel Kernel size in x-dimension.
 * @param[in] depth_kernel Kernel size in z-dimension.
 */
template <typename dtype>
__global__ void niblack_kernel_3d(dtype* image, float* output, float weight, int rows, int cols,
                                  int depth, int rows_kernel, int cols_kernel, int depth_kernel);

/**
 * @brief Host function to apply Niblack thresholding to a 3D image.
 *
 * @tparam dtype Input image type.
 * @param[in] image Input 3D image.
 * @param[out] output Output image after thresholding.
 * @param[in] weight Parameter for Niblack formula.
 * @param[in] rows Image height.
 * @param[in] cols Image width.
 * @param[in] depth Image depth.
 * @param[in] rows_kernel Kernel size in y-dimension.
 * @param[in] cols_kernel Kernel size in x-dimension.
 * @param[in] depth_kernel Kernel size in z-dimension.
 */
template <typename dtype>
void niblack_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
                       int rows_kernel, int cols_kernel, int depth_kernel);

/**
 * @brief GPU-accelerated Niblack thresholding over a full 3D volume.
 *
 * Works for small volumes that fit entirely into GPU memory.
 *
 * @tparam in_dtype Input image type.
 * @tparam out_dtype Output image type.
 * @param[in] hostImage Host-side input image pointer.
 * @param[out] hostOutput Host-side output image pointer.
 * @param[in] xsize Image width.
 * @param[in] ysize Image height.
 * @param[in] zsize Image depth.
 * @param[in] flag_verbose Verbosity flag.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 * @param[in] weight Niblack weight (typically negative for foreground emphasis).
 */
template <typename in_dtype, typename out_dtype>
void niblackThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,
                           int flag_verbose, int nx, int ny, int nz, float weight);

/**
 * @brief Chunked version of 3D Niblack thresholding.
 *
 * Suitable for large images that do not fit entirely into GPU memory.
 *
 * @tparam in_dtype Input image type.
 * @tparam out_dtype Output image type.
 * @param[in] hostImage Host-side input image pointer.
 * @param[out] hostOutput Host-side output image pointer.
 * @param[in] xsize Image width.
 * @param[in] ysize Image height.
 * @param[in] zsize Image depth.
 * @param[in] weight Niblack weight.
 * @param[in] type3d If true, apply full 3D kernel, otherwise 2D slice-wise.
 * @param[in] flag_verbose Verbosity flag.
 * @param[in] gpuMemory Fraction of GPU memory to use (0.0 to 1.0).
 * @param[in] ngpus Number of GPUs to use.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template<typename in_dtype, typename out_dtype>
void niblackThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,
                             float weight, int type3d, int flag_verbose,
                             float gpuMemory, int ngpus, int nx, int ny, int nz);

#endif  // NIBLACK_H