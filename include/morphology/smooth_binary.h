#ifndef BINARY_SMOOTH_H
#define BINARY_SMOOTH_H

#include "morphology.h"

/**
 * @brief Performs a sequence of binary morphological operations (opening and closing) on the GPU.
 *
 * This function executes a smoothing operation on a binary image by performing 
 * an alternating sequence of erosion and dilation operations. It uses CUDA to perform 
 * the computation on the device, leveraging memory management and optimized kernel calls.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param hostImage Pointer to the input image data on the host.
 * @param hostOutput Pointer to the output image data on the host.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param flag_verbose Verbosity flag for debugging and logging.
 * @param padding_bottom The padding size added at the bottom of the image.
 * @param padding_top The padding size added at the top of the image.
 * @param kernel Pointer to the morphological kernel on the host.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.8, 
 *       on page 682. The grayscale smoothing algorithm described was adapted for binary images.
 * @see morph_binary()
 */
template <typename dtype>
void smooth_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, const int flag_verbose, const int padding_bottom,
                             const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize);

/**
 * @brief Performs a sequence of binary morphological operations (opening and closing) on the CPU.
 *
 * This function executes a smoothing operation on a binary image by performing 
 * an alternating sequence of erosion and dilation operations. It runs entirely 
 * on the host CPU.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param hostImage Pointer to the input image data on the host.
 * @param hostOutput Pointer to the output image data on the host.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.8, 
 *       on page 682. The grayscale smoothing algorithm described was adapted for binary images.
 * @see morph_binary_on_host()
 */
template <typename dtype>
void smooth_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                           int kernel_zsize);

#endif  // BINARY_SMOOTH_H