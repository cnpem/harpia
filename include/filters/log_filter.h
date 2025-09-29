#ifndef LOG_FILTER_H
#define LOG_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/convolution.h"
#include "../common/kernels.h"

/**
 * @brief CUDA kernel for applying a 2D Laplacian of Gaussian (LoG) filter to a single slice of a 3D volume.
 *
 * This kernel processes a specific z-slice (`idz`) of the input volume using a 2D LoG kernel.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input image in device memory.
 * @param[out] output Pointer to the output image in device memory.
 * @param[in] deviceKernel Pointer to the LoG kernel in device memory.
 * @param[in] idz Index of the slice to process.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
template <typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz,
                                     int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel for applying a full 3D Laplacian of Gaussian (LoG) filter.
 *
 * This kernel applies the LoG filter across all dimensions of the 3D volume.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input 3D image in device memory.
 * @param[out] output Pointer to the output 3D image in device memory.
 * @param[in] deviceKernel Pointer to the 3D LoG kernel in device memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
template <typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* deviceKernel, int xsize,
                                     int ysize, int zsize);
/**
 * @brief Host function to apply LoG filtering to a 3D image.
 *
 * Depending on the `type` parameter, this function applies either 2D LoG filtering per slice or full 3D filtering.
 *
 * @tparam dtype Input image data type.
 * @param[in] image Pointer to the input image in host memory.
 * @param[out] output Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type Filtering mode: false for 2D, true for 3D.
 */
template <typename dtype>
void log_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type);

/**
 * @brief Chunked GPU execution of 3D LoG filtering for large images.
 *
 * This function handles large 3D images by filtering chunks of the volume, useful for memory-constrained environments.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @tparam kernel_dtype Kernel data type (e.g., float).
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] verbose Verbosity level (non-zero for detailed output).
 * @param[in] padding_bottom Amount of padding at the bottom of each chunk.
 * @param[in] padding_top Amount of padding at the top of each chunk.
 * @param[in] kernel Pointer to the LoG kernel in host memory.
 * @param[in] kernel_xsize Width of the kernel.
 * @param[in] kernel_ysize Height of the kernel.
 * @param[in] kernel_zsize Depth of the kernel.
 */
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void logFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize);

/**
 * @brief Chunked LoG filtering with optional multi-GPU support and automatic kernel management.
 *
 * This version automatically generates the LoG kernel and distributes the workload in chunks,
 * optionally across multiple GPUs.
 *
 * @tparam in_dtype Input image data type.
 * @tparam out_dtype Output image data type.
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type3d Filtering type: 0 for 2D, 1 for 3D.
 * @param[in] verbose Verbosity level.
 * @param[in] ngpus Number of GPUs to use (<= 0 for single GPU).
 * @param[in] safetyMargin Extra padding margin for chunk processing.
 */
template<typename in_dtype, typename out_dtype>
void logFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize, const int type3d,
                      const int verbose, int ngpus, const float safetyMargin);


#endif  // LOG_FILTER_H
