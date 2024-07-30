#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/common/grid_block_sizes.h"

#include <stdio.h>

/**
 * @brief Kernel function to perform pixel-wise subtraction of two images.
 * 
 * @tparam dtype Data type of the image.
 * @param deviceImage1 Pointer to the first input image on the device.
 * @param deviceImage2 Pointer to the second input image on the device.
 * @param deviceOutput Pointer to the output image on the device.
 * @param size Total number of pixels in the image.
 */
template<typename dtype>
__global__
void subtraction_pixel(dtype *deviceImage1, dtype *deviceImage2, dtype *deviceOutput, const int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) {
        deviceOutput[index] = deviceImage1[index] - deviceImage2[index];
    }
}
// Template instantiations for specific types
template __global__ void subtraction_pixel<u_int32_t>(u_int32_t*, u_int32_t*, u_int32_t*, const int);
template __global__ void subtraction_pixel<int>(int*, int*, int*, const int);
template __global__ void subtraction_pixel<float>(float*, float*, float*, const int);

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
template<typename dtype>
void subtraction(dtype *hostImage1, dtype *hostImage2, dtype *hostOutput, const int size, const int flag_verbose)
{
    // Set input dimension
    size_t nBytes = size * sizeof(dtype);

    // Malloc device global memory
    dtype *deviceImage1, *deviceImage2, *deviceOutput; 
    CHECK(cudaMalloc((dtype**)&deviceImage1, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceImage2, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));

    // Transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage1, hostImage1, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceImage2, hostImage2, nBytes, cudaMemcpyHostToDevice));

    // Set up execution configuration
    dim3 block(BLOCK_1D);
    dim3 grid((size + block.x - 1) / block.x);

    // Check grid and block dimension from host side
    if (flag_verbose) {
        printf("grid.x %d \n", grid.x);
        printf("block.x %d \n", block.x);
    }

    // Perform subtraction on the device
    subtraction_pixel<<<grid, block>>>(deviceImage1, deviceImage2, deviceOutput, size);
    cudaDeviceSynchronize(); // Ensure all GPU threads are finished

    // Transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(deviceImage1);
    cudaFree(deviceImage2);
    cudaFree(deviceOutput);
}
// Template instantiations for specific types
template void subtraction<u_int32_t>(u_int32_t *, u_int32_t *, u_int32_t *, const int, const int);
template void subtraction<int>(int *, int *, int *, const int, const int);
template void subtraction<float>(float *, float *, float *, const int, const int);

/**
 * @brief Perform pixel-wise subtraction of two images on the host.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage1 Pointer to the first input image on the host.
 * @param hostImage2 Pointer to the second input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param size Total number of pixels in the image.
 */
template<typename dtype>
void subtraction_on_host(dtype *hostImage1, dtype *hostImage2, dtype *hostOutput, const int size)
{
    for (int idx = 0; idx < size; idx++) {
        hostOutput[idx] = hostImage1[idx] - hostImage2[idx];
    }
}
// Template instantiations for specific types
template void subtraction_on_host<u_int32_t>(u_int32_t *, u_int32_t *, u_int32_t *, const int);
template void subtraction_on_host<int>(int *, int *, int *, const int);
template void subtraction_on_host<float>(float *, float *, float *, const int);