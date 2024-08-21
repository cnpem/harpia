#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/complement_binary.h"
#include <cuda_runtime.h>
#include <cstdint> // For uint16_t, unsigned int
#include <stdio.h>


template<typename dtype>
__global__
void complement_binary_kernel(dtype *deviceImage, dtype *deviceOutput, const int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        deviceOutput[index] =  1 - deviceImage[index];
    }
}
// Template instantiations for specific types
template __global__ void complement_binary_kernel<int>(int*, int*, const int);
template __global__ void complement_binary_kernel<unsigned int>(unsigned int*,unsigned int*, const int);
template __global__ void complement_binary_kernel<uint16_t>(uint16_t*, uint16_t*, const int);

template<typename dtype>
void complement_binary(dtype *deviceImage, dtype *deviceOutput, const int size, const int flag_verbose)
{
    
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
    cudaDeviceSynchronize(); // Ensure all GPU threads are finished

}
// Template instantiations for specific types
template void complement_binary<int>(int*, int*, const int, const int);
template void complement_binary<unsigned int>(unsigned int*,unsigned int*, const int, const int);
template void complement_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int);

template<typename dtype>
void complement_binary_on_device(dtype *hostImage, dtype *hostOutput, const int size, const int flag_verbose)
{
    
    // Set input dimension
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
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(deviceImage);
    cudaFree(deviceOutput);

}
// Template instantiations for specific types
template void complement_binary_on_device<int>(int*, int*, const int, const int);
template void complement_binary_on_device<unsigned int>(unsigned int*,unsigned int*, const int, const int);
template void complement_binary_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int);


template<typename dtype>
void complement_binary_on_host(dtype *hostImage, dtype *hostOutput, const int size)
{
    for (int index = 0; index < size; index++) {
        hostOutput[index] =  1 - hostImage[index];
    }
}
// Template instantiations for specific types
template void complement_binary_on_host<int>(int*, int*, const int);
template void complement_binary_on_host<unsigned int>(unsigned int*,unsigned int*, const int);
template void complement_binary_on_host<uint16_t>(uint16_t*,uint16_t*, const int);
