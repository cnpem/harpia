#ifndef LOGICAL_OR_H
#define LOGICAL_OR_H

/**
 * @brief Launch the CUDA kernel for logical OR operation.
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
void logical_or(dtype* deviceImage, dtype* deviceOutput, const size_t size,
                       const int flag_verbose);

/**
 * @brief Perform logical OR operation on an image using the GPU.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host.
 * @param hostOutput Output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Verbose flag to print execution details.
 * 
 * @see logical_or()
 */
template <typename dtype>
void logical_or_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, const int flag_verbose);

/**
 * @brief Perform logical OR operation on an image using the CPU.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host.
 * @param hostOutput Output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Verbose flag to print execution details.
 */
template <typename dtype>
void logical_or_on_host(dtype* hostImage, dtype* hostOutput, const size_t size);

#endif  // LOGICAL_OR_H