#include "../../include/common/grid_block_sizes.h"
#include "../../include/morphology/cuda_helper.h"

#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int

/**
 * @brief Kernel function to perform pixel-wise subtraction of two images.
 * 
 * @tparam dtype Data type of the image.
 * @param deviceImage Pointer to the first input image on the device.
 * @param deviceOutput Pointer to the output image on the device.
 * @param size Total number of pixels in the image.
 */
template <typename dtype>
__global__ void subtraction_kernel(dtype* deviceImage, dtype* deviceOutput, const int size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < size) {
    deviceOutput[index] = deviceOutput[index] - deviceImage[index];
  }
}
// Template instantiations for specific types
template __global__ void subtraction_kernel<int8_t>(int8_t*, int8_t*, const int);
template __global__ void subtraction_kernel<uint8_t>(uint8_t*, uint8_t*, const int);
template __global__ void subtraction_kernel<int16_t>(int16_t*, int16_t*, const int);
template __global__ void subtraction_kernel<uint16_t>(uint16_t*, uint16_t*, const int);
template __global__ void subtraction_kernel<unsigned int>(unsigned int*, unsigned int*, const int);
template __global__ void subtraction_kernel<int>(int*, int*, const int);
template __global__ void subtraction_kernel<float>(float*, float*, const int);

template <typename dtype>
void subtraction(dtype* deviceImage, dtype* deviceOutput, const int size, const int flag_verbose) {
  // Set up execution configuration
  dim3 block(BLOCK_1D);
  dim3 grid((size + block.x - 1) / block.x);

  // Check grid and block dimension from host side
  if (flag_verbose) {
    printf("grid.x %d \n", grid.x);
    printf("block.x %d \n", block.x);
  }

  // Perform subtraction on the device
  subtraction_kernel<<<grid, block>>>(deviceImage, deviceOutput, size);
  cudaDeviceSynchronize();  // Ensure all GPU threads are finished
}

// Template instantiations for specific types
template void subtraction<int8_t>(int8_t*, int8_t*, const int, const int);
template void subtraction<uint8_t>(uint8_t*, uint8_t*, const int, const int);
template void subtraction<int16_t>(int16_t*, int16_t*, const int, const int);
template void subtraction<uint16_t>(uint16_t*, uint16_t*, const int, const int);
template void subtraction<unsigned int>(unsigned int*, unsigned int*, const int, const int);
template void subtraction<int>(int*, int*, const int, const int);
template void subtraction<float>(float*, float*, const int, const int);

/**
 * @brief Perform pixel-wise subtraction of two images on the device.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage1 Pointer to the first input image on the host.
 * @param hostImage2 Pointer to the second input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param size Total number of pixels in the image.
 * @param flag_verbose Flag for verbose output.
 */
template <typename dtype>
void subtraction_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, const int flag_verbose) {
  // Set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Malloc device global memory
  dtype *deviceImage, *deviceOutput;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceOutput, hostOutput, nBytes, cudaMemcpyHostToDevice));

  // Perform subtraction on the device
  subtraction(deviceImage, deviceOutput, size, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}
// Template instantiations for specific types
template void subtraction_on_device<int8_t>(int8_t*, int8_t*, const int, const int, const int,
                                              const int);
template void subtraction_on_device<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int,
                                              const int);
template void subtraction_on_device<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                              const int);
template void subtraction_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                              const int);
template void subtraction_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                  const int, const int, const int);
template void subtraction_on_device<int>(int*, int*, const int, const int, const int, const int);
template void subtraction_on_device<float>(float*, float*, const int, const int, const int,
                                           const int);

/**
 * @brief Perform pixel-wise subtraction of two images on the host.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the first input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param size Total number of pixels in the image.
 */
template <typename dtype>
void subtraction_on_host(dtype* hostImage, dtype* hostOutput, const int size) {
  for (int idx = 0; idx < size; idx++) {
    hostOutput[idx] = hostOutput[idx] - hostImage[idx];
  }
}
// Template instantiations for specific types
template void subtraction_on_host<int8_t>(int8_t*, int8_t*, const int);
template void subtraction_on_host<uint8_t>(uint8_t*, uint8_t*, const int);
template void subtraction_on_host<int16_t>(int16_t*, int16_t*, const int);
template void subtraction_on_host<uint16_t>(uint16_t*, uint16_t*, const int);
template void subtraction_on_host<unsigned int>(unsigned int*, unsigned int*, const int);
template void subtraction_on_host<int>(int*, int*, const int);
template void subtraction_on_host<float>(float*, float*, const int);