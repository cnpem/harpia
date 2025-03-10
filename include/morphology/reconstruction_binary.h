#ifndef RECONSTRUCTION_BINARY_H
#define RECONSTRUCTION_BINARY_H

#include "morphology.h"

/**
 * @brief Performs morphological reconstruction (erosion or dilation) on an image using the GPU.
 *
 * This function operates entirely on the device, avoiding unnecessary memory transfers between
 * the host and the device. It iteratively applies geodesic erosion or dilation until convergence.
 * The results remain in device memory for further processing if needed.
 *
 * @tparam dtype The data type of the image.
 * @param deviceMarker Pointer to the marker image stored on the device (corresponds to the input image).
 * @param deviceMask Pointer to the mask image stored on the device.
 * @param deviceOutput Pointer to the output image stored on the device.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation to apply (EROSION or DILATION).
 * @param flag_verbose Flag to enable verbose output, printing grid and block dimensions.
 */
template <typename dtype>
void reconstruction_binary(dtype* deviceMarker, dtype* deviceMask, dtype* deviceOutput,
                           const int xsize, const int ysize, const int zsize, MorphOp operation,
                           const int flag_verbose);

/**
 * @brief Performs morphological reconstruction (erosion or dilation) on an image using the GPU,
 *        with memory management handled automatically.
 *
 * This function is called from the host. It allocates memory on the device, transfers the input data,
 * performs the reconstruction, and then transfers the result back to the host.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Pointer to the input image stored on the host (marker image).
 * @param hostMask Pointer to the mask image stored on the host.
 * @param hostOutput Pointer to the output image stored on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation to apply (EROSION or DILATION).
 * @param flag_verbose Flag to enable verbose output, printing grid and block dimensions.
 */                           
template <typename dtype>
void reconstruction_binary_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                     const int xsize, const int ysize, const int zsize,
                                     MorphOp operation, const int flag_verbose);

/**
 * @brief Performs morphological reconstruction (erosion or dilation) on an image using the CPU.
 *
 * This function runs entirely on the host and is primarily used for correctness verification of GPU results.
 * It iteratively applies geodesic erosion or dilation until convergence.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Pointer to the input image stored on the host (marker image).
 * @param hostMask Pointer to the mask image stored on the host.
 * @param hostOutput Pointer to the output image stored on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation to apply (EROSION or DILATION).
 */                                    
template <typename dtype>
void reconstruction_binary_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation);

#endif  // RECONSTRUCTION_BINARY_H