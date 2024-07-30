#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/subtraction.h"
#include "../../../include/common/grid_block_sizes.h"
#include <stdio.h>

/**
 * @brief Perform top-hat operation on the device.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the structuring element.
 * @param kernel_xsize Size of the structuring element in the x-dimension.
 * @param kernel_ysize Size of the structuring element in the y-dimension.
 * @param kernel_zsize Size of the structuring element in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
template<typename dtype>
void top_hat(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    // Set input dimension
    int size = xsize * ysize * zsize;    
    size_t nBytes = size * sizeof(dtype);

    // Set kernel dimension
    int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
    size_t kernel_nBytes = kernel_size * sizeof(int);

    // Malloc device global memory
    dtype *deviceImage, *deviceTmp, *deviceOutput; 
    int *deviceKernel; 
    CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
    CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

    // Transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

    // Set up execution configuration
    dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
    if (zsize == 1) block = dim3(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, (zsize + block.z - 1) / block.z);

    // Check grid and block dimension from host side
    if (flag_verbose) {
        printf("\nOpening operation configuration");
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    // Opening operation: erosion followed by dilation
    morph_grayscale<<<grid, block>>>(deviceImage, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, EROSION);
    cudaDeviceSynchronize(); // Assures all GPU threads are finished
    morph_grayscale<<<grid, block>>>(deviceOutput, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, DILATION);
    cudaDeviceSynchronize(); // Assures all GPU threads are finished

    // Set up execution configuration for subtraction
    dim3 block2(BLOCK_1D);
    dim3 grid2((size + block2.x - 1) / block2.x);

    // Check grid and block dimension from host side
    if (flag_verbose) {
        printf("\nSubtraction operation configuration\n");
        printf("grid.x %d grid.y %d grid.z %d\n", grid2.x, grid2.y, grid2.z);
        printf("block.x %d block.y %d block.z %d\n", block2.x, block2.y, block2.z);
    }

    // Top-hat: f - opening
    subtraction_pixel<<<grid2, block2>>>(deviceImage, deviceTmp, deviceOutput, size);
    cudaDeviceSynchronize(); // Assures all GPU threads are finished

    // Transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(deviceTmp);
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
// Template instantiations for specific types
template void top_hat<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void top_hat<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void top_hat<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);

/**
 * @brief Perform top-hat operation on the host.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the structuring element.
 * @param kernel_xsize Size of the structuring element in the x-dimension.
 * @param kernel_ysize Size of the structuring element in the y-dimension.
 * @param kernel_zsize Size of the structuring element in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
template<typename dtype>
void top_hat_on_host(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    // Set input dimension
    int size = xsize * ysize * zsize;    
    size_t nBytes = size * sizeof(dtype);

    // Allocate temporary memory
    dtype *host_tmp = (dtype *)malloc(nBytes);

    // Set input data
    memset(host_tmp, 0, nBytes); 

    // Opening operation
    MorphChain opening = {EROSION, DILATION};
    morph_chain_grayscale_on_host(hostImage, host_tmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, opening);

    // Top-hat: f - opening
    subtraction_on_host(hostImage, host_tmp, hostOutput, size);

    // Free temporary memory
    free(host_tmp);
}
// Template instantiations for specific types
template void top_hat_on_host<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void top_hat_on_host<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void top_hat_on_host<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);