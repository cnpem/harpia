#ifndef LBP_CUH
#define LBP_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief CUDA kernel to compute the Local Binary Pattern (LBP) of a 3D image slice.
 *
 * This kernel computes the LBP value for each pixel in the specified 2D slice (idz) of a 3D image.
 *
 * @tparam in_dtype Data type of the input image pixels.
 * @param[in] devImage Pointer to the device input image data.
 * @param[out] devOutput Pointer to the device output buffer for LBP results (float).
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image (number of slices).
 * @param[in] idz Index of the slice to process.
 */
template <typename in_dtype>
__global__ void lbp(in_dtype* devImage, float* devOutput, int xsize, int ysize, int zsize, int idz);

/**
 * @brief Host function to compute Local Binary Pattern for the entire 3D image.
 *
 * This function allocates device memory, launches the LBP kernel slice-by-slice, and copies the result back.
 *
 * @tparam in_dtype Data type of the input image.
 * @param[in] hostImage Pointer to the input image on the host.
 * @param[out] hostOutput Pointer to the output buffer on the host to store LBP results.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
template <typename in_dtype>
void localBinaryPattern(in_dtype* hostImage, float* hostOutput, int xsize, int ysize, int zsize);

#endif // LBP_CUH
