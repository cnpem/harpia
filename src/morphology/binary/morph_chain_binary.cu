#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/common/grid_block_sizes.h"
#include <stdio.h>


template<typename dtype>
void morph_chain_binary_on_device(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, MorphChain chain, const int flag_verbose){

    // set input dimension
    int size = xsize*ysize*zsize;    
    size_t nBytes = size * sizeof(dtype);

    // set kenrel dimension
    int kernel_size = kernel_xsize*kernel_ysize*kernel_zsize;
    size_t kernel_nBytes = kernel_size * sizeof(int);

    // malloc device global memory
    dtype *deviceImage, *deviceTmp ,*deviceOutput; 
    int *deviceKernel; 
    CHECK(cudaMalloc((int**)&deviceImage, nBytes));
    CHECK(cudaMalloc((int**)&deviceTmp, nBytes));
    CHECK(cudaMalloc((int**)&deviceOutput, nBytes));
    CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

    // transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

    //set up execution configuratio
    dim3 block (BLOCK_3D,BLOCK_3D,BLOCK_3D);
    dim3 grid((xsize+block.x-1)/block.x, (ysize+block.y-1)/block.y, (zsize+block.z-1)/block.z);

    // morphChain operation
    morph_binary_kernel<<<grid, block>>>(deviceImage, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, chain.operation1);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed
    morph_binary_kernel<<<grid, block>>>(deviceTmp, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, chain.operation2);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free host memorys
    cudaFree(deviceTmp);
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
template void morph_chain_binary_on_device<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain, const int);

template void morph_chain_binary_on_device<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, 
                                                  MorphChain, const int);
template void morph_chain_binary_on_device<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, 
                                                  MorphChain, const int);


//morphChain check operation on host
template<typename dtype>
void morph_chain_binary_on_host(dtype *hostImage, dtype *hostOutput, 
             int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
             const int xsize, const int ysize, const int zsize, MorphChain chain){

    // set input dimension
    int size = xsize*ysize*zsize;
    size_t nBytes = size * sizeof(dtype);

    // allocate temporary memory
    dtype *hostTmp;
    hostTmp = (dtype *)malloc(nBytes);

    // set input data
    memset(hostTmp, 0, nBytes); 
    
    // morphChain operation
    morph_binary_on_host(hostImage, hostTmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation1);
    morph_binary_on_host(hostTmp, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation2);

    //free temporary memory
    free(hostTmp);
}
template void morph_chain_binary_on_host<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morph_chain_binary_on_host<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morph_chain_binary_on_host<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, MorphChain);
