#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/common/grid_block_sizes.h"
#include <stdio.h>


template<typename dtype>
void morphChainGrayscaleOnDevice(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
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
    CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
    CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

    // transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

    //set up execution configuratio
    dim3 block(BLOCK_3D,BLOCK_3D,BLOCK_3D);
    if(zsize == 1) dim3 block(BLOCK_2D,BLOCK_2D,1);
    dim3 grid((xsize+block.x-1)/block.x, (ysize+block.y-1)/block.y, (zsize+block.z-1)/block.z);
   
    // check grid and block dimension from host side
    if(flag_verbose){
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    // morphChain operation
    morphGrayscaleKernel<<<grid, block>>>(deviceImage, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, chain.operation1);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed
    morphGrayscaleKernel<<<grid, block>>>(deviceTmp, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
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
template void morphChainGrayscaleOnDevice<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphChain, const int);
template void morphChainGrayscaleOnDevice<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain, const int);
template void morphChainGrayscaleOnDevice<float>(float *, float *, int *, int, int, int, const int, const int, const int, MorphChain, const int);


//morphChain check operation on host
template<typename dtype>
void morphChainGrayscaleOnHost(dtype *hostImage, dtype *hostOutput, 
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
    morphGrayscaleOnHost(hostImage, host_tmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation1);
    morphGrayscaleOnHost(host_tmp, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain.operation2);

    //free temporary memory
    free(host_tmp);
}
template void morphChainGrayscaleOnHost<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morphChainGrayscaleOnHost<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphChain);
template void morphChainGrayscaleOnHost<float>(float *, float *, int *, int, int, int, const int, const int, const int, MorphChain);
