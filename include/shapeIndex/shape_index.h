#ifndef GRADIENT2D_H
#define GRADIENT2D_H

#include <cuda_runtime.h>

/**
 * @brief CUDA kernel to compute the first-order finite difference gradient in 2D slices of a 3D volume.
 *
 * This kernel computes the gradient along a specified axis (x or y) for a given slice `idz`
 * of a 3D image. The finite difference approximation is used with edge handling.
 *
 * @tparam in_dtype Input image data type (e.g., int, unsigned int, float).
 * @param[in] devImage Pointer to the input image in device memory.
 * @param[out] devOutput Pointer to the output gradient array in device memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image (number of slices).
 * @param[in] idz Index of the slice to process (z-dimension).
 * @param[in] axis Gradient direction: 0 for x-axis, 1 for y-axis.
 * @param[in] step Finite difference step size (usually 1).
 */
template<typename in_dtype>
__global__ void gradient2D(in_dtype* devImage, float* devOutput,
                           int xsize, int ysize, int zsize,
                           int idz, int axis, float step);

/**
 * @brief Host function to compute gradients over all 2D slices of a 3D volume using CUDA.
 *
 * Allocates memory on the device, launches the `gradient2D` kernel for each slice, and copies
 * the result back to host memory. Supports gradient computation along x or y direction.
 *
 * @tparam in_dtype Input image data type (e.g., int, unsigned int, float).
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output gradient array in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image (number of slices).
 * @param[in] axis Gradient direction: 0 for x-axis, 1 for y-axis.
 * @param[in] step Finite difference step size (usually 1).
 */
template <typename in_dtype>
void gradient(in_dtype* hostImage, float* hostOutput,
              int xsize, int ysize, int zsize,
              int axis, float step);

#endif  // GRADIENT2D_H

