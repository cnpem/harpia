#ifndef NLMEANS_FILTER_H
#define NLMEANS_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Device function to compute the NL-means estimate for a pixel in 2D.
 *
 * For the target pixel at (idx, idy), this function calculates a weighted average of surrounding
 * patches using the NL-means algorithm, with Gaussian weighting and squared patch difference.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] image Pointer to the input image in device memory.
 * @param[out] mean Pointer to store the computed mean value.
 * @param[in] idx X-coordinate of the pixel.
 * @param[in] idy Y-coordinate of the pixel.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] small_window Size of the patch window (typically odd, like 3 or 5).
 * @param[in] big_window Size of the search window around the target pixel.
 * @param[in] h Filtering parameter that controls decay of weights.
 * @param[in] sigma Estimated standard deviation of noise.
 */
template <typename dtype>
__device__ void get_nlmean_kernel_2d(dtype* image, double* mean, int idx, int idy,
                                     int xsize, int ysize, int small_window, int big_window,
                                     double h, double sigma);

/**
 * @brief CUDA kernel to apply the NL-means filter over a 2D image.
 *
 * Each thread processes a pixel and computes the filtered value using patch similarity and
 * Gaussian weights over a search window.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] deviceImage Pointer to the input image in device memory.
 * @param[out] deviceOutput Pointer to the filtered output image in device memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] small_window Size of the local patch used for comparison.
 * @param[in] big_window Size of the search window around the pixel.
 * @param[in] h Filtering parameter (smoothing strength).
 * @param[in] sigma Estimated noise standard deviation.
 */
template <typename dtype>
__global__ void nlmeans_filter_kernel_2d(dtype* deviceImage, double* deviceOutput,
                                         int xsize, int ysize, int small_window,
                                         int big_window, double h, double sigma);

/**
 * @brief Host function to apply the NL-means filter on a 2D image.
 *
 * Allocates memory on the device, launches the filtering kernel, and copies results back.
 *
 * @tparam dtype Type of the input image data.
 * @param[in] hostImage Pointer to the input image in host memory.
 * @param[out] hostOutput Pointer to the output image in host memory.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] small_window Size of the patch window.
 * @param[in] big_window Size of the search window.
 * @param[in] h Filtering parameter (controls decay of weights).
 * @param[in] sigma Estimated standard deviation of the image noise.
 */
template <typename dtype>
void nlmeans_filtering(dtype* hostImage, double* hostOutput, int xsize, int ysize,
                       int small_window, int big_window, double h, double sigma);

#endif // NLMEANS_FILTER_H
