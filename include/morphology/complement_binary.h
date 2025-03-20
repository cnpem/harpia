#ifndef COMPLEMENT_BINARY_H
#define COMPLEMENT_BINARY_H

/**
 * @brief Launches the CUDA kernel to compute the binary complement of an image.
 * 
 * @tparam dtype Data type of the image.
 * @param deviceImage Pointer to input image in device memory (GPU).
 * @param deviceOutput Pointer to output image in device memory (GPU).
 * @param size Number of elements in the image.
 * @param flag_verbose Flag to enable verbose output for debugging.
 *
 * @see complement_binary_kernel()
 */
template <typename dtype>
void complement_binary(dtype* deviceImage, dtype* deviceOutput, const size_t size,
                       const int flag_verbose);

/**
 * @brief Performs the binary complement operation on an image using GPU.
 * 
 * Allocates memory on the GPU, transfers data, invokes the complement kernel,
 * and retrieves the results.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to input image in host memory (GPU).
 * @param hostOutput Pointer to output image in host memory (GPU).
 * @param xsize Image width.
 * @param ysize Image height.
 * @param zsize Image depth.
 * @param flag_verbose Flag to enable verbose output for debugging.
 *
 * @see complement_binary()
 */
template <typename dtype>
void complement_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, const int flag_verbose);

/**
 * @brief Computes the binary complement of an image on the host CPU.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to input image in host memory.
 * @param hostOutput Pointer to output image in host memory.
 * @param size Number of elements in the image.
 */
template <typename dtype>
void complement_binary_on_host(dtype* hostImage, dtype* hostOutput, const size_t size);

#endif  // COMPLEMENT_BINARY_H