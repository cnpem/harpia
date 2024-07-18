#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/subtraction.h"
#include <stdio.h>


template<typename dtype>
void topHat(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                 const int flag_verbose)
{

    // set input dimension
    int size = xsize*ysize*zsize;    
    size_t nBytes = size * sizeof(dtype);

    // set kenrel dimension
    int kernel_size = kernel_xsize*kernel_ysize*kernel_zsize;
    size_t kernel_nBytes = kernel_size * sizeof(int);

    // malloc device global memory
    dtype *deviceImage, *deviceTmp, *deviceOutput; 
    int *deviceKernel; 
    CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
    CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

    // transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

    //set up execution configuratio
    dim3 block(block_xsize,block_ysize,block_zsize);
    dim3 grid((xsize+block.x-1)/block.x, (ysize+block.y-1)/block.y, (zsize+block.z-1)/block.z);

    // check grid and block dimension from host side
    if(flag_verbose){
        printf("\nOpening operation configuration");
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    // opening operation
    morphGrayscaleKernel<<<grid, block>>>(deviceImage, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, EROSION);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed
    morphGrayscaleKernel<<<grid, block>>>(deviceOutput, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, DILATION);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    //set up execution configuratio
    dim3 block2(block_xsize);
    dim3 grid2((size+block.x-1)/block.x);

    // check grid and block dimension from host side
    if(flag_verbose){
        printf("\nSubtraction operation configuration\n");
        printf("grid.x %d grid.y %d grid.z %d\n", grid2.x, grid2.y, grid2.z);
        printf("block.x %d block.y %d block.z %d\n", block2.x, block2.y, block2.z);
    }

    //T_hat = f - opening
    subtractionKernel<<<grid2, block2>>>(deviceImage, deviceTmp, deviceOutput, xsize*ysize*zsize);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free host memorys
    cudaFree(deviceTmp);
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
template void topHat<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, 
                                const int);
template void topHat<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);
template void topHat<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);


template<typename dtype>
void topHatOnHost(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    // set input dimension
    int size = xsize*ysize*zsize;    
    size_t nBytes = size * sizeof(dtype);

    // allocate temporary memory
    // TODO: testar se n~ao precisa de tmp
    dtype *host_tmp;
    host_tmp = (dtype *)malloc(nBytes);

    // set input data
    memset(host_tmp, 0, nBytes); 

    // opening operation
    MorphChain opening = {EROSION, DILATION};
    morphChainGrayscaleOnHost(hostImage, host_tmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, opening);

    //T_hat = f - opening
    subtractionOnHost(hostImage, host_tmp, hostOutput, size);

    //free temporary memory
    free(host_tmp);
}
template void topHatOnHost<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void topHatOnHost<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void topHatOnHost<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);
