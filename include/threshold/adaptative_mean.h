#ifndef ADAPTATIVE_MEAN_H
#define ADAPTATIVE_MEAN_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"

/**
 * @brief Applies adaptive mean thresholding using a local mean filter.
 *
 * This function applies thresholding based on the local average of a neighborhood defined by the kernel sizes.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to input image data (flattened 3D array).
 * @param[out] output Pointer to output binary image data.
 * @param[in] weight Constant to subtract from the mean (threshold = local_mean - weight).
 * @param[in] rows Number of rows (height).
 * @param[in] cols Number of columns (width).
 * @param[in] depth Number of slices (depth).
 * @param[in] rows_kernel Height of the local kernel window.
 * @param[in] cols_kernel Width of the local kernel window.
 * @param[in] depth_kernel Depth of the local kernel window.
 */
template <typename dtype>
void local_mean_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel);

/**
 * @brief GPU-accelerated adaptive mean thresholding on full 3D volume.
 *
 * Suitable for small- to medium-sized images that fit entirely into GPU memory.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @param[in] hostImage Host-side input image pointer (flattened 3D).
 * @param[out] hostOutput Host-side output image pointer.
 * @param[in] xsize Width (columns).
 * @param[in] ysize Height (rows).
 * @param[in] zsize Depth (slices).
 * @param[in] flag_verbose Verbosity flag.
 * @param[in] nx Width of the kernel window.
 * @param[in] ny Height of the kernel window.
 * @param[in] nz Depth of the kernel window.
 * @param[in] weight Thresholding weight (subtracts from mean).
 */
template <typename in_dtype, typename out_dtype>
void adaptativeMeanThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                                   int nx, int ny, int nz, float weight);

/**
 * @brief Chunked GPU implementation of adaptive mean thresholding.
 *
 * Processes the image in chunks to reduce memory usage and allow execution on limited GPU resources.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @param[in] hostImage Host-side input image pointer.
 * @param[out] hostOutput Host-side output image pointer.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] weight Weight to subtract from the local mean.
 * @param[in] type3d If true, apply full 3D kernel; otherwise, apply 2D slice-by-slice.
 * @param[in] flag_verbose Verbosity level.
 * @param[in] gpuMemory Fraction of available GPU memory to use (e.g., 0.9 = 90%).
 * @param[in] ngpus Number of GPUs to utilize.
 * @param[in] nx Kernel window size in x-direction.
 * @param[in] ny Kernel window size in y-direction.
 * @param[in] nz Kernel window size in z-direction.
 */
template<typename in_dtype, typename out_dtype>
void adaptativeMeanThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, float weight,
                                     int type3d, int flag_verbose, float gpuMemory, int ngpus,
                                     int nx, int ny, int nz);

#endif  // ADAPTATIVE_MEAN_H