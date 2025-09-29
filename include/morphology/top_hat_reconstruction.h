#ifndef TOP_HAT_RECONSTRUCTION_H
#define TOP_HAT_RECONSTRUCTION_H

/**
 * @brief Computes a specialized top-hat transformation of a grayscale image on the GPU.
 *
 * The top-hat transformation enhances light structures in an image by computing the difference 
 * between the morphological opennig of the input image and the image itself. This implementation 
 * utilizes CUDA for efficient execution on the GPU.
 *
 * This version applies opennig by reconstruction instead of direct subtraction, preserving edge 
 * details. The openned image serves as a marker for grayscale reconstruction, where dilation 
 * progressively refines the output until convergence, ensuring accurate segmentation of porosity.
 *
 * @tparam dtype Data type of the image (e.g., int, unsigned int, float).
 * @param hostImage Pointer to the input image stored on the host.
 * @param hostOutput Pointer to the output image stored on the host.
 * @param xsize Width of the image in pixels.
 * @param ysize Height of the image in pixels.
 * @param zsize Depth of the image in pixels (for 3D processing).
 * @param flag_verbose Flag enabling verbose output for debugging (0 = silent, 1 = verbose).
 * @param kernel Pointer to the morphological kernel (structuring element).
 * @param kernel_xsize Width of the kernel in pixels.
 * @param kernel_ysize Height of the kernel in pixels.
 * @param kernel_zsize Depth of the kernel in pixels.
 *
 * @note This implementation is inspired by the Interactive Top-Hat by Reconstruction module,
 *       which enhances segmentation by applying grayscale reconstruction techniques.
 *       Reference: https://www.thermofisher.com/software-em-3d-vis/xtra-library/xtras/interactive-top-hat-by-reconstruction
 *
 * @see morph_grayscale(), reconstruction_grayscale(), subtraction()
 */
template <typename dtype>
void top_hat_reconstruction_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                      const int ysize, const int zsize, const int flag_verbose,
                                      int* kernel, int kernel_xsize, int kernel_ysize,
                                      int kernel_zsize);
/**
 * @brief Computes a specialized top-hat transformation of a grayscale image on the CPU.
 *
 * The top-hat transformation enhances light structures in an image by computing the difference 
 * between the morphological opennig of the input image and the image itself. This implementation 
 * is performed entirely on the host CPU.
 *
 * This version applies opennig by reconstruction instead of direct subtraction, preserving edge 
 * details. The openned image serves as a marker for grayscale reconstruction, where dilation 
 * progressively refines the output until convergence, ensuring accurate segmentation of porosity.
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
 * @note This implementation is inspired by the Interactive Top-Hat by Reconstruction module,
 *       which enhances segmentation by applying grayscale reconstruction techniques.
 *       Reference: https://www.thermofisher.com/software-em-3d-vis/xtra-library/xtras/interactive-top-hat-by-reconstruction
 *
 * @see morph_chain_grayscale_on_host(), reconstruction_grayscale_on_host(), subtraction_on_host()
 */
template <typename dtype>
void top_hat_reconstruction_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                    const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                    int kernel_ysize, int kernel_zsize);

#endif  // TOP_HAT_RECONSTRUCTION_H