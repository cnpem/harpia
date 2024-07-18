#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/cuda_helper.h"
#include <stdio.h>


template<typename dtype>
void morphChainBinaryOnDevice(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                        MorphChain chain, const int flag_verbose){
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
    dim3 block (block_xsize,block_ysize,block_zsize);
    dim3 grid((xsize+block.x-1)/block.x, (ysize+block.y-1)/block.y, (zsize+block.z-1)/block.z);

    // morphChain operation
    morphBinaryKernel<<<grid, block>>>(deviceImage, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, chain.operation1);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed
    morphBinaryKernel<<<grid, block>>>(deviceTmp, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
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
template void morphChainBinaryOnDevice<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                                        const int, MorphChain, const int);
template void morphChainBinaryOnDevice<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, 
                                                        const int, const int, const int, MorphChain, const int);
template void morphChainBinaryOnDevice<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, 
                                                        const int, const int, const int, MorphChain, const int);


//morphChain check operation on host
template<typename dtype>
void morphChainBinaryOnHost(dtype *hostImage, dtype *hostOutput, 
             int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
             const int xsize, const int ysize, const int zsize, MorphChain chain){

    // set input dimension
    int size = xsize*ysize*zsize;
    size_t nBytes = size * sizeof(dtype);

    // allocate temporary memory
    dtype *host_tmp;
    host_tmp = (dtype *)malloc(nBytes);

    // set input data
    memset(host_tmp, 0, nBytes); 
    
    // morphChain operation
    morphBinaryOnHost(hostImage, host_tmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation1);
    morphBinaryOnHost(host_tmp, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation2);

    //free temporary memory
    free(host_tmp);
}
template void morphChainBinaryOnHost<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morphChainBinaryOnHost<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morphChainBinaryOnHost<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, MorphChain);
