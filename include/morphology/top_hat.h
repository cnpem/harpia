#ifndef TOP_HAT_H
#define TOP_HAT_H

/**
 * @brief Computes the top-hat transformation of a grayscale image on the GPU.
 *
 * The top-hat transformation enhances light structures in an image by computing the difference 
 * between the input image and it's morphological opennig. This implementation utilizes CUDA for 
 * efficient execution on the GPU.
 *
 * The morphological operations (erosion followed by dilation) are performed using a 3D structuring 
 * element (kernel).  Additional padding is applied to handle border effects when large input images 
 * are divided into chunks to fit within the available GPU memory.
 *
 * @tparam dtype Data type of the image (e.g., int, unsigned int, float).
 * @param hostImage Pointer to the input image stored on the host.
 * @param hostOutput Pointer to the output image stored on the host.
 * @param xsize Width of the image in pixels.
 * @param ysize Height of the image in pixels.
 * @param zsize Depth of the image in pixels (for 3D processing).
 * @param flag_verbose Flag enabling verbose output for debugging (0 = silent, 1 = verbose).
 * @param padding_bottom Number of padding layers added at the top of the image.
 * @param padding_top Number of padding layers added at the top of the image.
 * @param kernel Pointer to the morphological kernel (structuring element).
 * @param kernel_xsize Width of the kernel in pixels.
 * @param kernel_ysize Height of the kernel in pixels.
 * @param kernel_zsize Depth of the kernel in pixels.
 *
 * @note This implementation follows the morphological transformation principles described in:
 *       R.C. Gonzalez, R.E. Woods, "Digital Image Processing", 4th Edition, Pearson, 2018.
 *       Chapter 9 (Morphological Image Processing), Section 9.8, pages 683-685.
 *
 * @see morph_grayscale(), subtraction()
 */
template <typename dtype>
void top_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, const int padding_bottom,
                       const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                       int kernel_zsize);

/**
 * @brief Computes the top-hat transformation of a grayscale image on the CPU.
 *
 * The top-hat transformation enhances light structures in an image by computing the difference 
 * between the input image and it's morphological opennig. This implementation is performed entirely 
 * on the host CPU.
 *
 * The transformation applies a sequence of morphological operations (erosion followed by dilation) 
 * using a 3D structuring element (kernel).
 *
 * @tparam dtype Data type of the image (e.g., int, unsigned int, float).
 * @param hostImage Pointer to the input image stored on the host.
 * @param hostOutput Pointer to the output image stored on the host.
 * @param xsize Width of the image in pixels.
 * @param ysize Height of the image in pixels.
 * @param zsize Depth of the image in pixels (for 3D processing).
 * @param kernel Pointer to the morphological kernel (structuring element).
 * @param kernel_xsize Width of the kernel in pixels.
 * @param kernel_ysize Height of the kernel in pixels.
 * @param kernel_zsize Depth of the kernel in pixels.
 *
 * @note This implementation follows the morphological transformation principles described in:
 *       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
 *       Chapter 9 (Morphological Image Processing), Section 9.8, pages 683-685.
 *
 * @see morph_chain_grayscale_on_host(), subtraction_on_host()
 */ 
template <typename dtype>
void top_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                     const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                     int kernel_zsize);

#endif  // TOP_HAT_H