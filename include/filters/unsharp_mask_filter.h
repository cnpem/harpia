#ifndef UNSHARP_MASK_FILTER_H
#define UNSHARP_MASK_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "gaussian_filter.h"

/**
 * @brief Applies an Unsharp Mask filter to a 2D or 3D image.
 *
 * This function sharpens the image by subtracting a blurred version of it (Gaussian-filtered)
 * and adding the difference back scaled by an amplification factor (amount).
 * The formula is:  
 * \f[
 *     \text{output} = \text{image} + \text{amount} \cdot (\text{image} - \text{blurred})
 * \f]
 * Thresholding can be applied to ignore small differences.
 *
 * @tparam dtype Data type of the input image (e.g., float, unsigned char).
 * @param[in] image Pointer to the input image on host.
 * @param[out] output Pointer to the output sharpened image on host.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image (number of slices).
 * @param[in] sigma Standard deviation of the Gaussian blur.
 * @param[in] ammount Amplification factor for the high-pass component.
 * @param[in] threshold Threshold below which differences are ignored.
 * @param[in] type If true, applies 3D Gaussian filtering; otherwise, applies it slice-by-slice (2D).
 */
template <typename dtype>
void unsharp_mask_filtering(dtype* image, float* output, int xsize, int ysize, int zsize,
                            float sigma, float ammount, float threshold, bool type);


/**
 * @brief Chunked GPU version of the Unsharp Mask filter for large 3D images.
 *
 * This function divides the image volume into chunks to fit in GPU memory, then applies the
 * unsharp masking process in parallel using one or more GPUs.
 *
 * @tparam dtype Data type of the input image.
 * @param[in] image Pointer to the input image on host.
 * @param[out] output Pointer to the output sharpened image on host.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] sigma Standard deviation for the Gaussian blur.
 * @param[in] ammount Sharpening factor.
 * @param[in] threshold Threshold for ignoring small differences.
 * @param[in] type3d If true, performs 3D filtering; otherwise, 2D per slice.
 * @param[in] verbose Verbosity flag for logging or profiling.
 * @param[in] ngpus Number of GPUs to use for processing.
 * @param[in] safetyMargin Fraction of GPU memory to leave unused (between 0.0 and 1.0).
 */
template <typename dtype>
void unsharpMaskChunked(dtype* image, float* output, int xsize, int ysize, int zsize,
                        float sigma, float ammount, float threshold, const int type3d, const int verbose, int ngpus,
                        const float safetyMargin);

#endif  // UNSHARP_MASK_FILTER_H
