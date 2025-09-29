#ifndef FILL_HOLES_H
#define FILL_HOLES_H

#include "morphology.h"

/**
 * @brief Performs hole filling on a 3D binary image using morphological reconstruction on the GPU.
 *
 * This function transfers a 3D binary image to the GPU, creates a marker image to identify holes,
 * and performs hole filling using morphological reconstruction. The filled image is then copied 
 * back to the host.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag to enable verbose output for debugging.
 * 
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.6, 
 *       on pages 671-672.
 * @see fill_holes_marker(), complement_binary(), reconstruction_binary()
 */ 
 
template <typename dtype>
void fill_holes_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, const int flag_verbose);

/**
 * @brief Fills holes in a binary image using CPU operations.
 *
 * This function performs hole filling using morphological reconstruction on a binary image.
 *
 * @tparam dtype Data type of the image (e.g., uint8_t, int16_t, etc.).
 * @param hostImage Pointer to the input binary image.
 * @param hostOutput Pointer to the output filled image.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image (set to 1 for 2D images).
 * 
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.6, 
 *       on pages 671-672.
 * @see reconstruction_binary_on_host(), complement_binary_on_host()
 */ 
template <typename dtype>
void fill_holes_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize);

#endif  // FILL_HOLES_H