#ifndef COMPARE_ARRAYS_GRAYSCALE_H
#define COMPARE_ARRAYS_GRAYSCALE_H

#include "morphology.h"

/**
 * @brief Launches a CUDA kernel to compare two grayscale device arrays element-wise.
 * 
 * This function configures the CUDA grid and block dimensions, then calls the 
 * `compare_arrays_grayscale_kernel` function to compare `deviceImage1` and `deviceImage2`. 
 * The result is stored in `deviceOutput`.
 *
 * @tparam dtype The data type of the elements in the arrays.
 * @param deviceImage1 Pointer to the first input array on the device (GPU).
 * @param deviceImage2 Pointer to the second input array on the device (GPU).
 * @param deviceOutput Pointer to the output flag on the device (GPU).
 * @param size The total number of elements (pixels) in the arrays.
 * @param flag_verbose Flag to enable verbose output for debugging.
 *
 * @see compare_arrays_grayscale_kernel()
 */
template <typename dtype>
void compare_arrays_grayscale(dtype* deviceImage1, dtype* deviceImage2, int* deviceOutput,
                              const size_t size, const int flag_verbose);

/**
 * @brief Performs element-wise comparison of two grayscale arrays on the GPU using CUDA.
 * 
 * This function allocates memory on the GPU, copies input data from host to device, calls 
 * `compare_arrays_grayscale` to perform the comparison, and then copies the result back to the host.
 *
 * @tparam dtype The data type of the elements in the arrays.
 * @param hostImage1 Pointer to the first input array on the host (CPU).
 * @param hostImage2 Pointer to the second input array on the host (CPU).
 * @param hostOutput Pointer to the output flag on the host (CPU).
 * @param size The total number of elements (pixels) in the arrays.
 * @param flag_verbose If nonzero, prints debugging information about memory allocation and copying.
 *
 * @see compare_arrays_grayscale()
 */
template <typename dtype>
void compare_arrays_grayscale_on_device(dtype* hostImage1, dtype* hostImage2, int* hostOutput,
                                        const size_t size, const int flag_verbose);

/**
 * @brief Function to perform pixel-wise comparison to check if two grayscale arrays are equal on the host 
 * (CPU). 
 * 
 * This function compares two input arrays (`hostImage1` and `hostImage2`) element by element. If 
 * any corresponding elements are not equal, the output flag (`hostOutput`) is set to false, and the 
 * function exits early.
 * 
 * @tparam dtype The data type of the elements in the arrays.
 * @param hostImage1 Pointer to the first input array on the host (CPU).
 * @param hostImage2 Pointer to the second input array on the host (CPU).
 * @param hostOutput Pointer to the output flag on the host (CPU).
 * @param size The total number of elements (pixels) in the arrays.
 */
template <typename dtype>
void compare_arrays_grayscale_on_host(dtype* hostImage1, dtype* hostImage2, int* hostOutput,
                                      const size_t size);

#endif  // COMPARE_ARRAYS_GRAYSCALE_H