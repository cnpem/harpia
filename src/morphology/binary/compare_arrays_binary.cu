#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/compare_arrays_binary.h"
#include "../../../include/morphology/cuda_helper.h"

/**
 * @brief Kernel function to perform pixel-wise comparison to check if two arrays are equal on the 
 * GPU. 
 * 
 * This function compares two input arrays (`deviceImage1` and `deviceImage2`) element by element. 
 * If any corresponding elements are not equal, the output flag (`deviceOutput`) is set to 0 
 * (false). The comparison is performed in parallel using CUDA threads.
 *
 * @tparam dtype The data type of the elements in the arrays.
 * @param deviceImage1 Pointer to the first input array on the device (GPU).
 * @param deviceImage2 Pointer to the second input array on the device (GPU).
 * @param deviceOutput Pointer to the output flag on the device (GPU). Should be 
 * initialized to `1` before calling this function.
 * @param size The total number of elements (pixels) in the arrays.
 */
template <typename dtype>
__global__ void compare_arrays_binary_kernel(dtype* deviceImage1, dtype* deviceImage2,
                                             int* deviceOutput, const size_t size) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    if (deviceImage1[index] != deviceImage2[index]) {
      atomicAnd(deviceOutput, 0);
      /**
      * @note The atomic operation is used to set `deviceOutput` to `false` in a thread-safe 
      * manner. This ensures that only one thread modifies `deviceOutput` at a time, 
      * preventing race conditions.
      */
    }
  }
}
template __global__ void compare_arrays_binary_kernel<int>(int*, int*, int*, const size_t);
template __global__ void compare_arrays_binary_kernel<unsigned int>(unsigned int*, unsigned int*,
                                                                    int*, const size_t);
template __global__ void compare_arrays_binary_kernel<int16_t>(int16_t*, int16_t*, int*, const 
                                                               size_t);
template __global__ void compare_arrays_binary_kernel<uint16_t>(uint16_t*, uint16_t*, int*,
                                                                const size_t);

template <typename dtype>
void compare_arrays_binary(dtype* deviceImage1, dtype* deviceImage2, int* deviceOutput,
                           const size_t size, const int flag_verbose) {

  // Set up execution configuration
  dim3 block(BLOCK_1D);
  dim3 grid((size + block.x - 1) / block.x);

  // Check grid and block dimension from host side
  if (flag_verbose) {
    printf("grid.x %d \n", grid.x);
    printf("block.x %d \n", block.x);
  }

  // Perform operation on the device
  compare_arrays_binary_kernel<<<grid, block>>>(deviceImage1, deviceImage2, deviceOutput, size);
  cudaDeviceSynchronize();  // Ensure all GPU threads are finished
}
// Template instantiations for specific types
template void compare_arrays_binary<int>(int*, int*, int*, const size_t, const int);
template void compare_arrays_binary<unsigned int>(unsigned int*, unsigned int*, int*, const size_t,
                                                  const int);
template void compare_arrays_binary<int16_t>(int16_t*, int16_t*, int*, const size_t, const int);
template void compare_arrays_binary<uint16_t>(uint16_t*, uint16_t*, int*, const size_t, const int);

template <typename dtype>
void compare_arrays_binary_on_device(dtype* hostImage1, dtype* hostImage2, int* hostOutput,
                                     const size_t size, const int flag_verbose) {

  // Set input dimension
  size_t nBytes = size * sizeof(dtype);

  // Malloc device global memory
  dtype *deviceImage1, *deviceImage2;
  int* deviceOutput;
  CHECK(cudaMalloc((dtype**)&deviceImage1, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceImage2, nBytes));
  CHECK(cudaMalloc((int**)&deviceOutput, nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage1, hostImage1, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceImage2, hostImage2, nBytes, cudaMemcpyHostToDevice));

  // Perform operation on the device
  compare_arrays_binary(deviceImage1, deviceImage2, deviceOutput, size, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(deviceImage1);
  cudaFree(deviceImage2);
  cudaFree(deviceOutput);
}
template void compare_arrays_binary_on_device<int>(int*, int*, int*, const size_t, const int);
template void compare_arrays_binary_on_device<unsigned int>(unsigned int*, unsigned int*, int*,
                                                            const size_t, const int);
template void compare_arrays_binary_on_device<int16_t>(int16_t*, int16_t*, int*, const size_t,
                                                       const int);
template void compare_arrays_binary_on_device<uint16_t>(uint16_t*, uint16_t*, int*, const size_t,
                                                        const int);

template <typename dtype>
void compare_arrays_binary_on_host(dtype* hostImage1, dtype* hostImage2, int* hostOutput,
                                   const size_t size) {
  *hostOutput = 1;
  for (size_t index = 0; index < size; index++) {
    if (hostImage1[index] != hostImage2[index]) {
      *hostOutput = 0;
      return;  // Exit on first mismatch
    }
  }
}
template void compare_arrays_binary_on_host<int>(int*, int*, int*, const size_t);
template void compare_arrays_binary_on_host<unsigned int>(unsigned int*, unsigned int*, int*,
                                                          const size_t);
template void compare_arrays_binary_on_host<int16_t>(int16_t*, int16_t*, int*, const size_t);
template void compare_arrays_binary_on_host<uint16_t>(uint16_t*, uint16_t*, int*, const size_t);
