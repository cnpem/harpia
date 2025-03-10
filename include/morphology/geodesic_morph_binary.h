#ifndef GEODESIC_BINARY_MORPHOLOGY_H
#define GEODESIC_BINARY_MORPHOLOGY_H

#include "morphology.h"

/**
 * @brief Launches the CUDA kernel for geodesic morphological operations.
 *
 * This function sets up the kernel execution configuration and launches the CUDA kernel.
 *
 * @tparam dtype The data type of the image.
 * @param deviceImage Input image on the GPU.
 * @param deviceMask Mask image on the GPU.
 * @param deviceOutput Output image on the GPU.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image.
 * @param flag_verbose If nonzero, prints debugging information about kernel execution.
 * @param padding_bottom Padding at the bottom in the z-dimension.
 * @param padding_top Padding at the top in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
void geodesic_morph_binary(dtype* deviceImage, dtype* deviceMask, dtype* deviceOutput,
                           const int xsize, const int ysize, const int zsize,
                           const int flag_verbose, const int padding_bottom, const int padding_top,
                           MorphOp operation);


/**
 * @brief Perform geodesic erosion/dilation operation on the entire image using the GPU.
 * This function is called from the host and slides the `morph_binary` kernel function
 * through all pixels of the input image.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 * @param padding_bottom Number of padding layers added at the bottom of the image.
 * @param padding_top Number of padding layers added at the top of the image.
 * 
 * @return None. The function modifies `hostOutput` in place.
 */
template <typename dtype>
void geodesic_morph_binary_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                     const int xsize, const int ysize, const int zsize,
                                     const int flag_verbose, const int padding_bottom,
                                     const int padding_top, MorphOp operation);


/**
 * @brief Perform geodesic erosion/dilation operation on the entire image using the CPU.
 * This function is used to verify the correctness of the GPU results by computing
 * the same operation on the host.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 *
 * @return None. The function modifies `hostOutput` in place.
 */
template <typename dtype>
void geodesic_morph_binary_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation);

#endif  // GEODESIC_BINARY_MORPHOLOGY_H