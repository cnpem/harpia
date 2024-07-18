#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/cuda_helper.h"
#include <stdio.h>

// Perform erosin operation for one pixel
// If the value is 1 -> 255 is saved in the pixel output
// If the value is 0 -> 0 is saved in the pixel output

template<typename dtype>
CUDA_HOSTDEV 
void morphPixelGrayscale(dtype *image, dtype *output, 
                             int centerIdx, int centerIdy, int centerIdz, 
                             int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                             const int xsize, const int ysize, const int zsize, MorphOp operation)
{
    dtype *im = image;
    int *ik = kernel;

    //initialize auxiliar value with the central pixel
    dtype aux = im[centerIdz*xsize*ysize + centerIdy*xsize + centerIdx];

    int imageIdx, imageIdy, imageIdz, index;

    int startIdx = centerIdx - kernel_xsize/2;
    int startIdy = centerIdy - kernel_ysize/2;
    int startIdz = centerIdz - kernel_zsize/2;

    for (int iz = 0; iz < kernel_zsize; iz ++){
        for (int iy = 0; iy < kernel_ysize; iy ++){
            for (int ix = 0 ; ix < kernel_xsize; ix ++){

                imageIdx =  startIdx + ix;
                imageIdy =  startIdy + iy;
                imageIdz =  startIdz + iz;
                index = imageIdz*xsize*ysize + imageIdy*xsize + imageIdx;

                // ignore out of bounds pixels and don't care pixels
                if(imageIdx < 0 || imageIdx > xsize-1 || 
                   imageIdy < 0 || imageIdy > ysize-1 || 
                   imageIdz < 0 || imageIdz > zsize-1 ||
                   ik[ix] < 0){
                    // do nothing.
                }

                else{
                    if(operation == EROSION){
                        aux = (im[index] < aux) ? im[index] : aux; //erosion: aux is the min value
                    } 
                    else{ 
                        aux = (im[index] > aux) ? im[index] : aux; //dilation: aux is the max value 
                    }
                }
            }
        }
    }
    output[centerIdz*ysize*xsize + centerIdy*xsize + centerIdx] = aux;
}
template CUDA_HOSTDEV void morphPixelGrayscale<u_int32_t>(u_int32_t*, u_int32_t*, int, int, int, int*, int, int, int, 
                                                                 const int, const int, const int, MorphOp);
template CUDA_HOSTDEV void morphPixelGrayscale<int>(int*, int*, int, int, int, int*, int, int, int, 
                                                               const int, const int, const int, MorphOp);
template CUDA_HOSTDEV void morphPixelGrayscale<float>(float*, float*, int, int, int, int*, int, int, int, 
                                                                 const int, const int, const int, MorphOp);

template<typename dtype>
__global__
void morphGrayscaleKernel(dtype *deviceImage, dtype *deviceOutput, 
                       int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                       const int xsize, const int ysize, const int zsize, MorphOp operation)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idz = threadIdx.z + blockIdx.z*blockDim.z;

    if (idx < xsize && idy < ysize && idz < zsize){
        morphPixelGrayscale(deviceImage, deviceOutput, idx, idy, idz, 
                        kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                        xsize, ysize, zsize, operation);
    }
}        
template __global__ void morphGrayscaleKernel<u_int32_t>(u_int32_t*, u_int32_t*, int*, int, int, int, const int, const int, const int, MorphOp);
template __global__ void morphGrayscaleKernel<int>(int*, int*, int*, int, int, int, const int, const int, const int, MorphOp);
template __global__ void morphGrayscaleKernel<float>(float*, float*, int*, int, int, int, const int, const int, const int, MorphOp);

// Slide kernel and erosion operation over all input image pixels
template<typename dtype>
void morphGrayscaleOnDevice(dtype *hostImage, dtype *hostOutput, 
                            int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                            const int xsize, const int ysize, const int zsize, const int block_xsize, 
                            const int block_ysize, const int block_zsize, MorphOp operation, const int flag_verbose)
{
    // set input dimension
    int size = xsize*ysize*zsize;    
    size_t nBytes = size * sizeof(dtype);

    // set kenrel dimension
    int kernel_size = kernel_xsize*kernel_ysize*kernel_zsize;
    size_t kernel_nBytes = kernel_size * sizeof(int);

    // malloc device global memory
    dtype *deviceImage, *deviceOutput; 
    int *deviceKernel; 
    CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
    CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
    CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

    // transfer data from the host to the device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

    //set up execution configuratio
    dim3 block (block_xsize,block_ysize,block_zsize);
    dim3 grid((xsize+block.x-1)/block.x, (ysize+block.y-1)/block.y, (zsize+block.z-1)/block.z);

    // check grid and block dimension from host side
    if(flag_verbose){
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    } 

    // device erosion/dialation 
    morphGrayscaleKernel<<<grid, block>>>(deviceImage, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                          xsize, ysize, zsize, operation);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free host memorys
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
template void morphGrayscaleOnDevice<u_int32_t>(u_int32_t*, u_int32_t*, int*, int, int, int, const int, const int, const int, const int, const int, const int, MorphOp, const int);
template void morphGrayscaleOnDevice<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, MorphOp, const int);
template void morphGrayscaleOnDevice<float>(float*, float*, int*, int, int, int, const int, const int, const int, const int, const int, const int, MorphOp, const int);

// Slide kernel and erosion operation over all input image pixels
template<typename dtype>
void morphGrayscaleOnHost(dtype *hostImage, dtype *hostOutput, 
                       int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                       const int xsize, const int ysize, const int zsize, MorphOp operation)
{
    for(int idz = 0; idz < zsize; idz++){
        for(int idy = 0; idy < ysize; idy++){
            for(int idx = 0; idx < xsize; idx++){

                morphPixelGrayscale(hostImage, hostOutput, idx, idy, idz, 
                                kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                                xsize, ysize, zsize, operation);
            }
        }
    }// slide over image
}
template void morphGrayscaleOnHost<u_int32_t>(u_int32_t*, u_int32_t*, int*, int, int, int, const int, const int, const int, MorphOp);
template void morphGrayscaleOnHost<int>(int*, int*, int*, int, int, int, const int, const int, const int, MorphOp);
template void morphGrayscaleOnHost<float>(float*, float*, int*, int, int, int, const int, const int, const int, MorphOp);
