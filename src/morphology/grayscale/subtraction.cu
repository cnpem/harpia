#include "../../../include/morphology/cuda_helper.h"

#include <stdio.h>

//kernel for image subtraction
template<typename dtype>
__global__
void subtractionKernel(dtype *deviceImage1,dtype *deviceImage2, dtype *deviceOutput, const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;

    if (index < size){
        deviceOutput[index] = deviceImage1[index] - deviceImage2[index];
    }
}        
template __global__ void subtractionKernel<u_int32_t>(u_int32_t*, u_int32_t*, u_int32_t*, const int);
template __global__ void subtractionKernel<int>(int*, int*, int*, const int);
template __global__ void subtractionKernel<float>(float*, float*, float*, const int);


// Slide kernel and erosion operation over all input image pixels
template<typename dtype>
void subtraction(dtype *hostImage1,dtype *hostImage2, dtype *hostOutput, const int size, 
                         const int block_size, const int flag_verbose)
{
    // set input dimension
    size_t nBytes = size * sizeof(dtype);

    // malloc device global memory
    dtype *deviceImage1, *deviceImage2, *deviceOutput; 
    CHECK(cudaMalloc((dtype**)&deviceImage1, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceImage2, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));

    // transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage1, hostImage1, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceImage2, hostImage2, nBytes, cudaMemcpyHostToDevice));

    //set up execution configuratio
    dim3 block (block_size);
    dim3 grid((size+block.x-1)/block.x);

    // check grid and block dimension from host side
    if(flag_verbose){
        printf("grid.x %d \n", grid.x);
        printf("block.x %d \n", block.x);
    } 

    // device erosion/dialation 
    subtractionKernel<<<grid, block>>>(deviceImage1, deviceImage2, deviceOutput, size);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free host memorys
    cudaFree(deviceImage1);
    cudaFree(deviceImage2);
    cudaFree(deviceOutput);
}
template void subtraction<u_int32_t>(u_int32_t *,u_int32_t *, u_int32_t *, const int, const int, const int);
template void subtraction<int>(int *,int *, int *, const int, const int, const int);
template void subtraction<float>(float *,float *, float *, const int, const int, const int);


// Slide kernel and erosion operation over all input image pixels
template<typename dtype>
void subtractionOnHost(dtype *hostImage1,dtype *hostImage2, dtype *hostOutput, const int size)
{
    for(int idx = 0; idx < size; idx++){
        hostOutput[idx] = hostImage1[idx] - hostImage2[idx];
    }// slide over image
}
template void subtractionOnHost<u_int32_t>(u_int32_t *,u_int32_t *, u_int32_t *, const int);
template void subtractionOnHost<int>(int *,int *, int *, const int);
template void subtractionOnHost<float>(float *,float *, float *, const int);