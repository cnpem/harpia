#ifndef MEAN_FILTER_H
#define MEAN_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief CUDA device function to compute the mean over a 2D neighborhood.
 *
 * Computes the mean intensity value in a (nx × ny) region centered at (i, j).
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to input image in device memory.
 * @param[out] mean Pointer to output mean value.
 * @param[in] i Row index of the center pixel.
 * @param[in] j Column index of the center pixel.
 * @param[in] xsize Width of the image.
 * @param[in] cols Total number of columns in the image (may be equal to xsize).
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 */
template <typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int xsize, int cols,
                                   int nx, int ny);

/**
 * @brief CUDA device function to compute the mean over a 3D neighborhood.
 *
 * Computes the mean intensity value in a (nx × ny × nz) region centered at (i, j, k).
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to input image in device memory.
 * @param[out] mean Pointer to output mean value.
 * @param[in] i X index.
 * @param[in] j Y index.
 * @param[in] k Z index.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template <typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean, int i, int j, int k, int xsize,
                                   int ysize, int zsize, int nx, int ny, int nz);

/**
 * @brief CUDA kernel to apply a 2D mean filter to a single slice of a 3D image.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to input 3D image in device memory.
 * @param[out] output Pointer to output 2D slice in device memory.
 * @param[in] xsize Width of the slice.
 * @param[in] ysize Height of the slice.
 * @param[in] idz Index of the slice to process.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 */
template <typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int xsize, int ysize, int idz,
                                      int nx, int ny);

/**
 * @brief CUDA kernel to apply a 3D mean filter to an entire volume.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to input 3D image in device memory.
 * @param[out] output Pointer to output 3D image in device memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template <typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output, int xsize, int ysize, int zsize,
                                      int nx, int ny, int nz);

/**
 * @brief Host function to apply a mean filter to a 3D image.
 *
 * This function chooses between 2D or 3D filtering based on the value of `nz`.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input image in host memory.
 * @param[out] output Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template <typename dtype>
void mean_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, int nx, int ny,
                    int nz);

/**
 * @brief Chunked version of 3D mean filtering for large datasets (single GPU).
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] flag_verbose Verbosity flag (0 = silent).
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template <typename in_dtype, typename out_dtype>
void meanFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                     int nx, int ny, int nz);

/**
 * @brief Chunked multi-GPU version of mean filtering for very large datasets.
 *
 * Automatically splits the image into memory-safe chunks and applies mean filtering.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type3d Flag to select between 2D (0) or 3D (1) filtering.
 * @param[in] flag_verbose Verbosity flag.
 * @param[in] gpuMemory Fraction of GPU memory to use (0.0 to 1.0).
 * @param[in] ngpus Number of GPUs to use.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template<typename in_dtype, typename out_dtype>
void meanFilterChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int type3d, int flag_verbose,
                       float gpuMemory, int ngpus, int nx, int ny, int nz);

#endif  // MEAN_FILTER_H