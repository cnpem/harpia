#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/subtraction.h"
#include <stdio.h>


template<typename dtype>
void bottomHat(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
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
        printf("\nClosing operation configuration\n");
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    // closing operation
    morphGrayscaleKernel<<<grid, block>>>(deviceImage, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, DILATION);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed
    morphGrayscaleKernel<<<grid, block>>>(deviceOutput, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, EROSION);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    //set up execution configuratio
    dim3 block2(block_xsize);
    dim3 grid2((size+block.x-1)/block.x);

    // check grid and block dimension from host side
    if(flag_verbose){
        printf("\nSubtraction operation configuration");
        printf("grid.x %d grid.y %d grid.z %d\n", grid2.x, grid2.y, grid2.z);
        printf("block.x %d block.y %d block.z %d\n", block2.x, block2.y, block2.z);
    }

    //B_hat = closing - f
    subtractionKernel<<<grid2, block2>>>(deviceTmp, deviceImage, deviceOutput, xsize*ysize*zsize);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free host memorys
    cudaFree(deviceTmp);
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
template void bottomHat<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, const int);
template void bottomHat<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                   const int, const int);
template void bottomHat<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int, const int, 
                               const int, const int);


template<typename dtype>
void bottomHatOnHost(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    // set input dimension
    int size = xsize*ysize*zsize;    
    size_t nBytes = size * sizeof(dtype);

    // allocate temporary memory
    // TODO: testar se não precisa de tmp
    dtype *host_tmp;
    host_tmp = (dtype *)malloc(nBytes);

    // set input data
    memset(host_tmp, 0, nBytes); 

    // opening operation
    MorphChain closing = {DILATION, EROSION};
    morphChainGrayscaleOnHost(hostImage, host_tmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, closing);

    //B_hat = closimg - f
    subtractionOnHost(host_tmp, hostImage, hostOutput, size);

    //free temporary memory
    free(host_tmp);
}
template void bottomHatOnHost<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void bottomHatOnHost<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void bottomHatOnHost<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);
