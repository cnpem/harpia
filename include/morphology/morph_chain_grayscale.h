#ifndef GRAYSCALE_MORPHOLOGY_CHAIN_H
#define GRAYSCALE_MORPHOLOGY_CHAIN_H

#include "morphology.h"

/**
 * @brief Performs closing/opening operation on a grayscale image using the GPU.
 *
 * This function applies a sequence of erosion and dilation operations, commonly used for 
 * morphological opening or closing, to a grayscale image on the GPU. It supports 3D images and 
 * configurable kernel sizes.
 *
 * @tparam dtype Data type of the image (e.g., int, uint8_t, etc.).
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Width of the image (x-dimension).
 * @param ysize Height of the image (y-dimension).
 * @param zsize Depth of the image (z-dimension).
 * @param flag_verbose Flag to enable verbose output for debugging.
 * @param padding_bottom Number of padding pixels added to the bottom of the image.
 * @param padding_top Number of padding pixels added to the top of the image.
 * @param kernel Pointer to the morphological kernel (structuring element).
 * @param kernel_xsize Width of the kernel (x-dimension).
 * @param kernel_ysize Height of the kernel (y-dimension).
 * @param kernel_zsize Depth of the kernel (z-dimension).
 * @param chain MorphChain structure containing the operations to be performed (erosion/dilation).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.8, 
 *       on pages 680-682.
 * @see morph_grayscale()
 */
template <typename dtype>
void morph_chain_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                     const int ysize, const int zsize, const int flag_verbose,
                                     const int padding_bottom, const int padding_top, int* kernel,
                                     int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                     MorphChain chain);

/**
 * @brief Performs a closing/opening operation on a grayscale image using the CPU.
 *
 * This function applies a sequence of erosion and dilation operations, commonly used for 
 * morphological opening or closing, to a grayscale image on the CPU. It supports 3D images and 
 * configurable kernel sizes.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param chain MorphChain structure containing the operations to be performed (erosion/dilation).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.8, 
 *       on pages 680-682.
 * @see morph_grayscale_on_host()
 */
template <typename dtype>
void morph_chain_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                   const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                   int kernel_ysize, int kernel_zsize, MorphChain chain);

#endif  // GRAYSCALE_MORPHOLOGY_CHAIN_H