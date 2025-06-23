#ifndef SAUVOLA_H
#define SAUVOLA_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"

/**
 * @brief CUDA kernel for 2D Sauvola thresholding.
 *
 * Applies the Sauvola formula to each pixel in a 2D image slice:
 * `T = mean * (1 + weight * ((std / range) - 1))`
 *
 * @tparam dtype Input image type.
 * @param[in] image Input image.
 * @param[out] output Output thresholded image.
 * @param[in] weight Tuning parameter (commonly around 0.5).
 * @param[in] range Value range of dtype (e.g., 255 for 8-bit images).
 * @param[in] rows Number of rows in the image.
 * @param[in] cols Number of columns in the image.
 * @param[in] idz Slice index to process.
 * @param[in] rows_kernel Kernel height.
 * @param[in] cols_kernel Kernel width.
 */
template <typename dtype>
__global__ void sauvola_kernel_2d(dtype* image, float* output, float weight, dtype range, int rows,
                                  int cols, int idz, int rows_kernel, int cols_kernel);

/**
 * @brief CUDA kernel for 3D Sauvola thresholding.
 *
 * Applies Sauvola's adaptive thresholding formula to a full 3D volume.
 *
 * @tparam dtype Input image type.
 * @param[in] image Input 3D image.
 * @param[out] output Output thresholded volume.
 * @param[in] weight Tuning parameter.
 * @param[in] range Value range of dtype.
 * @param[in] rows Image height.
 * @param[in] cols Image width.
 * @param[in] depth Image depth.
 * @param[in] rows_kernel Kernel height.
 * @param[in] cols_kernel Kernel width.
 * @param[in] depth_kernel Kernel depth.
 */
template <typename dtype>
__global__ void sauvola_kernel_3d(dtype* image, float* output, float weight, dtype range, int rows,
                                  int cols, int depth, int rows_kernel, int cols_kernel,
                                  int depth_kernel);

/**
 * @brief Host function to apply Sauvola thresholding to a 3D image.
 *
 * Performs thresholding using mean and standard deviation in a local 3D window.
 *
 * @tparam dtype Input image type.
 * @param[in] image Input image pointer.
 * @param[out] output Output image pointer.
 * @param[in] weight Sauvola weight factor.
 * @param[in] range Maximum value of input dtype.
 * @param[in] rows Image height.
 * @param[in] cols Image width.
 * @param[in] depth Image depth.
 * @param[in] rows_kernel Kernel height.
 * @param[in] cols_kernel Kernel width.
 * @param[in] depth_kernel Kernel depth.
 */
template <typename dtype>
void sauvola_threshold(dtype* image, float* output, float weight, dtype range, int rows, int cols,
                       int depth, int rows_kernel, int cols_kernel, int depth_kernel);

/**
 * @brief GPU-based Sauvola thresholding for full 3D volumes.
 *
 * Suitable for images that fit entirely in GPU memory.
 *
 * @tparam in_dtype Input image type.
 * @tparam out_dtype Output image type.
 * @param[in] hostImage Input image on host.
 * @param[out] hostOutput Output image on host.
 * @param[in] xsize Width of image.
 * @param[in] ysize Height of image.
 * @param[in] zsize Depth of image.
 * @param[in] flag_verbose Verbose flag for diagnostics.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 * @param[in] weight Sauvola tuning parameter.
 * @param[in] range Max pixel value of input type (e.g. 255 for uint8).
 */
template <typename in_dtype, typename out_dtype>
void sauvolaThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,
                           int flag_verbose, int nx, int ny, int nz, float weight, in_dtype range);

/**
 * @brief Chunked Sauvola thresholding for large 3D volumes.
 *
 * Splits the image into chunks to handle large datasets that don't fit in GPU memory.
 *
 * @tparam in_dtype Input image type.
 * @tparam out_dtype Output image type.
 * @param[in] hostImage Input image on host.
 * @param[out] hostOutput Output image on host.
 * @param[in] xsize Image width.
 * @param[in] ysize Image height.
 * @param[in] zsize Image depth.
 * @param[in] weight Sauvola tuning parameter.
 * @param[in] range Max pixel value of input type.
 * @param[in] type3d Whether to use full 3D filtering or 2D slice-wise.
 * @param[in] flag_verbose Verbose flag.
 * @param[in] gpuMemory GPU memory usage fraction (0.0–1.0).
 * @param[in] ngpus Number of GPUs to use.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template<typename in_dtype, typename out_dtype>
void sauvolaThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,
                             float weight, in_dtype range, int type3d, int flag_verbose,
                             float gpuMemory, int ngpus, int nx, int ny, int nz);

#endif  // SAUVOLA_H