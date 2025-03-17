#ifndef BINARY_MORPHOLOGY_H
#define BINARY_MORPHOLOGY_H

#include "morphology.h"

/**
 * @brief Apply a binary morphological operation (erosion or dilation) to an entire image using the GPU.
 * 
 * This function configures and launches a CUDA kernel (`morph_binary_kernel`) to process an entire 
 * 3D image using a morphological operation. The function avoids unnecessary memory transfers 
 * between the host and device by operating entirely on the GPU. The result remains in device memory.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param deviceImage Pointer to the input image stored in GPU memory.
 * @param deviceOutput Pointer to the output image stored in GPU memory.
 * @param xsize Width of the image (number of pixels in the x-dimension).
 * @param ysize Height of the image (number of pixels in the y-dimension).
 * @param zsize Depth of the image (number of pixels in the z-dimension).
 * @param flag_verbose Flag to enable verbose output for debugging (prints grid and block dimensions).
 * @param padding_bottom Number of padding layers at the bottom of the image.
 * @param padding_top Number of padding layers at the top of the image.
 * @param deviceKernel Pointer to the structuring element (kernel) used for the operation.
 * @param kernel_xsize Width of the kernel (x-dimension).
 * @param kernel_ysize Height of the kernel (y-dimension).
 * @param kernel_zsize Depth of the kernel (z-dimension).
 * @param operation Type of morphological operation (EROSION or DILATION).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.2, 
 *       on pages 638-643.
 * @see morph_binary_kernel()
 */
template <typename dtype>
void morph_binary(dtype* deviceImage, dtype* deviceOutput, const int xsize, const int ysize,
                  const int zsize, const int flag_verbose, const int padding_bottom,
                  const int padding_top, int* deviceKernel, int kernel_xsize, int kernel_ysize,
                  int kernel_zsize, MorphOp operation);

/**
 * @brief Perform binary erosion or dilation on an entire image using the GPU.
 *
 * This function is called from the host. It allocates memory on the device, transfers the input image
 * and kernel data to the GPU, performs the specified morphological operation (erosion or dilation),
 * and then copies the result back to the host.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param flag_verbose If nonzero, prints details about kernel execution configuration.
 * @param padding_bottom The padding size added at the bottom of the image.
 * @param padding_top The padding size added at the top of the image.
 * @param hostKernel Pointer to the morphological kernel on the host.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param operation The morphological operation to apply (EROSION or DILATION).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.2, 
 *       on pages 638-643.
 * @see morph_binary()
 */
 template <typename dtype>
void morph_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, const int flag_verbose, const int padding_bottom,
                            const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                            int kernel_zsize, MorphOp operation);

/**
 * @brief Perform binary morphological operations (erosion or dilation) on the host (CPU).
 *
 * This function applies a binary morphological operation (either erosion or dilation)
 * directly on the host CPU. It does not use the GPU. The function iterates over the input
 * image and applies the specified kernel element-wise to determine the output values.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param hostImage Pointer to the input image data on the host.
 * @param hostOutput Pointer to the output image data on the host.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param hostKernel Pointer to the morphological kernel on the host.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param operation The morphological operation to apply (EROSION or DILATION).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.2, 
 *       on pages 638-643.
 * @see morph_binary_pixel()
 */
template <typename dtype>
void morph_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                          int kernel_zsize, MorphOp operation);

#endif  // BINARY_MORPHOLOGY_H