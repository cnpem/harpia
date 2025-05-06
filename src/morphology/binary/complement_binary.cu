#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream> // for debug cout code
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/complement_binary.h"
#include "../../../include/morphology/cuda_helper.h"

/**
 * @brief CUDA kernel to compute the binary complement of an image.
 * 
 * @tparam dtype Data type of the image (e.g., int, uint16_t, etc.).
 * @param deviceImage Pointer to input image in device memory.
 * @param deviceOutput Pointer to output image in device memory.
 * @param size Number of elements in the image.
 */
template <typename dtype>
__global__ void complement_binary_kernel(dtype* deviceImage, dtype* deviceOutput, const size_t size) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    deviceOutput[index] = 1 - deviceImage[index];
  }
}
template __global__ void complement_binary_kernel<int>(int*, int*, const size_t);
template __global__ void complement_binary_kernel<unsigned int>(unsigned int*, unsigned int*,
                                                                const size_t);
template __global__ void complement_binary_kernel<int16_t>(int16_t*, int16_t*, const size_t);
template __global__ void complement_binary_kernel<uint16_t>(uint16_t*, uint16_t*, const size_t);

template <typename dtype>
void complement_binary(dtype* deviceImage, dtype* deviceOutput, const size_t size,
                       const int flag_verbose) {

  // Set up execution configuration
  dim3 block(BLOCK_1D);
  dim3 grid((size + block.x - 1) / block.x);

  // Check grid and block dimension from host side
  if (flag_verbose) {
    printf("grid.x %d \n", grid.x);
    printf("block.x %d \n", block.x);
  }

  // Perform subtraction on the device
  complement_binary_kernel<<<grid, block>>>(deviceImage, deviceOutput, size);
  cudaDeviceSynchronize();  // Ensure all GPU threads are finished
}
template void complement_binary<int>(int*, int*, const size_t, const int);
template void complement_binary<unsigned int>(unsigned int*, unsigned int*, const size_t, const int);
template void complement_binary<int16_t>(int16_t*, int16_t*, const size_t, const int);
template void complement_binary<uint16_t>(uint16_t*, uint16_t*, const size_t, const int);

template <typename dtype>
void complement_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, const int flag_verbose) {

  // DEBUG
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (flag_verbose) std::cout << "Free GPU memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Malloc device global memory
  dtype *deviceImage, *deviceOutput;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(deviceOutput, 0, nBytes));  //Initialize output image to zero

  // Perform subtraction on the device
  complement_binary(deviceImage, deviceOutput, size, flag_verbose);

  // Transfer data from the device to the host
  //DEBUG
  if (flag_verbose) {
    printf("before cudaMemcpy()\n");
    std::cout << "cudaMemcpy parameters:" << std::endl;
    std::cout << "  hostOutput (dst): " << static_cast<void*>(hostOutput) << std::endl;
    std::cout << "  deviceOutput (src): " << static_cast<void*>(deviceOutput) << std::endl;
    std::cout << "  nBytes: " << nBytes << " (" << (nBytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "  direction: cudaMemcpyDeviceToHost" << std::endl;
  }
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost)); //original code
  if (flag_verbose) printf("after cudaMemcpy()\n\n");

  // Free device memory
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}
template void complement_binary_on_device<int>(int*, int*, const int, const int, const int,
                                               const int);
template void complement_binary_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                        const int, const int, const int);
template void complement_binary_on_device<int16_t>(int16_t*, int16_t*, const int, const int,
                                                   const int, const int);
template void complement_binary_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                    const int, const int);

template <typename dtype>
void complement_binary_on_host(dtype* hostImage, dtype* hostOutput, const size_t size) {
  for (size_t index = 0; index < size; index++) {
    hostOutput[index] = 1 - hostImage[index];
  }
}
template void complement_binary_on_host<int>(int*, int*, const size_t);
template void complement_binary_on_host<unsigned int>(unsigned int*, unsigned int*, const size_t);
template void complement_binary_on_host<int16_t>(int16_t*, int16_t*, const size_t);
template void complement_binary_on_host<uint16_t>(uint16_t*, uint16_t*, const size_t);
