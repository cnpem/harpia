#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include <cuda_runtime.h>
#include <chrono>

/**
 * @brief Device function to sort an array using bubble sort.
 *
 * Used to find the median value in a neighborhood window.
 *
 * @tparam dtype Data type of the elements to sort.
 * @param[in,out] array Pointer to the array in device memory.
 * @param[in] size Size of the array.
 */
template <typename dtype>
__device__ void bubble_sort(dtype* array, int size);

/**
 * @brief Device function to extract the neighborhood and compute the median in 2D.
 *
 * Collects values in a (nx × ny) neighborhood centered at (i, j) from a 2D slice.
 *
 * @tparam dtype Image data type.
 * @param[in] image Input image in device memory.
 * @param[out] kernel Buffer to store the neighborhood values.
 * @param[in] i Row index of the pixel.
 * @param[in] j Column index of the pixel.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] nx Neighborhood width.
 * @param[in] ny Neighborhood height.
 */
template <typename dtype>
__device__ void get_median_kernel_2d(dtype* image, dtype* kernel, int i, int j, int xsize,
                                     int ysize, int nx, int ny);

/**
 * @brief Device function to extract the neighborhood and compute the median in 3D.
 *
 * Collects values in a (nx × ny × nz) neighborhood centered at (i, j, k).
 *
 * @tparam dtype Image data type.
 * @param[in] image Input 3D image in device memory.
 * @param[out] kernel Buffer to store the neighborhood values.
 * @param[in] i X index.
 * @param[in] j Y index.
 * @param[in] k Z index.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Neighborhood width.
 * @param[in] ny Neighborhood height.
 * @param[in] nz Neighborhood depth.
 */
template <typename dtype>
__device__ void get_median_kernel_3d(dtype* image, dtype* kernel, int i, int j, int k, int xsize,
                                     int ysize, int zsize, int nx, int ny, int nz);

/**
 * @brief CUDA kernel to apply 2D median filtering to a single slice of a 3D image.
 *
 * Each thread processes a single pixel in the 2D slice at index `idz`.
 *
 * @tparam dtype Image data type.
 * @param[in] image Input 3D image in device memory.
 * @param[out] output Output slice in device memory.
 * @param[out] kernel Temporary storage for neighborhood values.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] idz Z-slice index.
 * @param[in] nx Neighborhood width.
 * @param[in] ny Neighborhood height.
 */
template <typename dtype>
__global__ void median_filter_kernel_2d(dtype* image, dtype* output, dtype* kernel, int xsize,
                                        int ysize, int idz, int nx, int ny);

/**
 * @brief CUDA kernel to apply 3D median filtering.
 *
 * Each thread processes a single voxel in the 3D volume.
 *
 * @tparam dtype Image data type.
 * @param[in] image Input 3D image in device memory.
 * @param[out] output Output 3D image in device memory.
 * @param[out] kernel Temporary storage for neighborhood values.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] idz Index of the current block in the z-dimension (can be ignored if not chunked).
 * @param[in] nx Neighborhood width.
 * @param[in] ny Neighborhood height.
 * @param[in] nz Neighborhood depth.
 */
template <typename dtype>
__global__ void median_filter_kernel_3d(dtype* image, dtype* output, dtype* kernel, int xsize,
                                        int ysize, int zsize, int idz, int nx, int ny, int nz);

/**
 * @brief Host function to apply a 2D or 3D median filter.
 *
 * Automatically selects 2D or 3D kernel based on nz.
 *
 * @tparam dtype Image data type.
 * @param[in] image Pointer to the input image in host memory.
 * @param[out] output Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] nx Neighborhood width.
 * @param[in] ny Neighborhood height.
 * @param[in] nz Neighborhood depth.
 */
template <typename dtype>
void median_filtering(dtype* image, dtype* output, int xsize, int ysize, int zsize, int nx, int ny,
                      int nz);

#endif  // MEDIAN_FILTER_H