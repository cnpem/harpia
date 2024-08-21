#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/common/grid_block_sizes.h"
#include <stdio.h>

/**
 * @brief Performs a chain of grayscale morphological operations on the device.
 * 
 * This function applies a series of morphological operations defined in a `MorphChain` on an image using
 * CUDA. It allocates memory on the device, transfers data, applies the operations, and then copies the 
 * result back to the host.
 * 
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param hostImage Pointer to the input grayscale image on the host.
 * @param hostOutput Pointer to the output image where the result will be stored on the host.
 * @param kernel Pointer to the kernel used for the morphological operations.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param chain A `MorphChain` structure containing the sequence of operations to be performed.
 * @param flag_verbose If non-zero, print verbose output about the grid and block dimensions.
 */
template<typename dtype>
void morph_chain_grayscale_on_device(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, MorphChain chain, const int flag_verbose){
    // set input dimension
    int size = xsize * ysize * zsize;    
    size_t nBytes = size * sizeof(dtype);

    // set kernel dimension
    int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
    size_t kernel_nBytes = kernel_size * sizeof(int);

    // malloc device global memory
    dtype *deviceImage, *deviceTmp ,*deviceOutput; 
    int *deviceKernel; 
    CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
    CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

    // transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

    // Perform the first operation in the chain
    morph_grayscale(deviceImage, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, chain.operation1, flag_verbose);

    // Perform the second operation in the chain
    morph_grayscale(deviceTmp, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, chain.operation2, flag_verbose);

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free device memory
    cudaFree(deviceTmp);
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
template void morph_chain_grayscale_on_device<unsigned int>(unsigned int *, unsigned int *, int *, int, int, int, const int, const int, const int, MorphChain, const int);
template void morph_chain_grayscale_on_device<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain, const int);
template void morph_chain_grayscale_on_device<float>(float *, float *, int *, int, int, int, const int, const int, const int, MorphChain, const int);


/**
 * @brief Performs a chain of grayscale morphological operations on the host.
 * 
 * This function applies a sequence of morphological operations defined in a `MorphChain` on an image using
 * a CPU-based approach. It first allocates temporary memory, applies the operations sequentially, and 
 * then frees the temporary memory.
 * 
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param hostImage Pointer to the input grayscale image on the host.
 * @param hostOutput Pointer to the output image where the result will be stored on the host.
 * @param kernel Pointer to the kernel used for the morphological operations.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param chain A `MorphChain` structure containing the sequence of operations to be performed.
 */
template<typename dtype>
void morph_chain_grayscale_on_host(dtype *hostImage, dtype *hostOutput, 
             int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
             const int xsize, const int ysize, const int zsize, MorphChain chain){

    // set input dimension
    int size = xsize * ysize * zsize;
    size_t nBytes = size * sizeof(dtype);

    // allocate temporary memory
    dtype *hostTmp;
    hostTmp = (dtype *)malloc(nBytes);

    // initialize temporary memory
    memset(hostTmp, 0, nBytes); 
    
    // Perform the first operation in the chain
    morph_grayscale_on_host(hostImage, hostTmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation1);

    // Perform the second operation in the chain
    morph_grayscale_on_host(hostTmp, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation2);

    // Free temporary memory
    free(hostTmp);
}
template void morph_chain_grayscale_on_host<unsigned int>(unsigned int *, unsigned int *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morph_chain_grayscale_on_host<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morph_chain_grayscale_on_host<float>(float *, float *, int *, int, int, int, const int, const int, const int, MorphChain);