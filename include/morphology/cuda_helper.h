#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>

/**
 * @brief Checks for CUDA errors and prints an error message if any.
 * 
 * @param error The CUDA error code.
 * @param file The file name where the error occurred.
 * @param line The line number where the error occurred.
 */
void throw_on_cuda_error(cudaError_t error, const char* file, int line);

/**
 * @brief Tests and prints the information about the CUDA device.
 */
void test_check_device_info();

/**
 * @brief Checks and prints the GPU memory usage.
 *
 * This function retrieves the current free and total memory available on the GPU,
 * computes the used memory, and prints these values along with the specified 
 * allocated memory in gigabytes (GB).
 *
 * @param allocatedBytes The amount of memory (in bytes) allocated by the program.
 *
 * @note This function uses `cudaMemGetInfo()` to query GPU memory details and prints 
 *       the memory usage in a human-readable format.
 */
void checkGpuMem(size_t allocatedBytes);

#define CHECK(call) \
  { throw_on_cuda_error((call), __FILE__, __LINE__); };
#endif  //CUDA_HELPER_H