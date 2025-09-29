#ifndef SUBTRACTION_H
#define SUBTRACTION_H

#include "morphology.h"

/**
 * @brief Launch the CUDA kernel for subtraction operation.
 *
 * @tparam dtype The data type of the image.
 * @param deviceImage Input image on the device.
 * @param deviceOutput Output image on the device.
 * @param size Total number of elements in the image.
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 * 
 * @see logical_or_kernel()
 */
template <typename dtype>
void subtraction(dtype* deviceImage, dtype* deviceOutput, const size_t size, const int flag_verbose);

/**
 * @brief Perform pixel-wise subtraction of two images on the device.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage1 Pointer to the first input image on the host.
 * @param hostImage2 Pointer to the second input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Width of the image (number of pixels in the x-dimension).
 * @param ysize Height of the image (number of pixels in the y-dimension).
 * @param zsize Depth of the image (number of pixels in the z-dimension).
 * @param flag_verbose Flag for verbose output.
 * 
 * @see subtraction()
 */
template <typename dtype>
void subtraction_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, const int flag_verbose);

/**
 * @brief Perform pixel-wise subtraction of two images on the host.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the first input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param size Total number of pixels in the image.
 */
template <typename dtype>
void subtraction_on_host(dtype* hostImage, dtype* hostOutput, const size_t size);

#endif  // SUBTRACTION_H