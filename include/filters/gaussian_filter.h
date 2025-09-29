#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief CUDA kernel for 2D Gaussian filtering on a single slice of a 3D volume.
 *
 * Applies a 2D Gaussian filter using the specified kernel to the slice `idz` of a 3D image.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input image in device memory.
 * @param[out] output Pointer to the output image in device memory.
 * @param[in] deviceKernel Pointer to the Gaussian kernel in device memory.
 * @param[in] idz Index of the slice (z-dimension) to process.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 */
template <typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, double* deviceKernel, int idz,
                                          int xsize, int ysize, int zsize, int nx, int ny);

/**
 * @brief CUDA kernel for full 3D Gaussian filtering on a volume.
 *
 * Applies a 3D Gaussian filter using the specified kernel across all dimensions.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input 3D image in device memory.
 * @param[out] output Pointer to the output 3D image in device memory.
 * @param[in] deviceKernel Pointer to the 3D Gaussian kernel in device memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Kernel width.
 * @param[in] ny Kernel height.
 * @param[in] nz Kernel depth.
 */
template <typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, double* deviceKernel,
                                          int xsize, int ysize, int zsize, int nx, int ny, int nz);

/**
 * @brief Host function to perform Gaussian filtering on a 3D image.
 *
 * Depending on the `type` parameter, this function either applies 2D filtering slice-by-slice
 * or full 3D filtering across all dimensions.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input image in host memory.
 * @param[out] output Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] sigma Standard deviation for the Gaussian kernel.
 * @param[in] type Filtering mode: false for 2D, true for 3D.
 */
template <typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma,
                        bool type);



/**
 * @brief Chunked GPU Gaussian filter execution for large 3D images.
 *
 * This version splits the image into chunks and applies 3D filtering to each chunk,
 * allowing better memory control and performance for large data.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @tparam kernel_dtype Kernel data type (e.g., double).
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] verbose Verbosity flag (non-zero for logging).
 * @param[in] padding_bottom Amount of padding below the chunk.
 * @param[in] padding_top Amount of padding above the chunk.
 * @param[in] kernel Pointer to the 3D Gaussian kernel.
 * @param[in] kernel_xsize Width of the kernel.
 * @param[in] kernel_ysize Height of the kernel.
 * @param[in] kernel_zsize Depth of the kernel.
 */
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void gaussianFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize);
/**
 * @brief Chunked Gaussian filtering with automatic kernel generation and multi-GPU support.
 *
 * Handles large 3D images by dividing them into chunks and optionally distributing the workload
 * across multiple GPUs. The kernel is generated based on the input `sigma`.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] sigma Standard deviation for the Gaussian kernel.
 * @param[in] type3d Filtering mode: 0 for 2D, 1 for 3D.
 * @param[in] verbose Verbosity flag.
 * @param[in] ngpus Number of GPUs to use (<= 0 for single GPU).
 * @param[in] safetyMargin Extra margin to ensure safe chunking.
 */                   
template<typename in_dtype, typename out_dtype>
void gaussianFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize, float sigma, const int type3d,
                      const int verbose, int ngpus,const float safetyMargin);

#endif  // GAUSSIAN_FILTER_H
