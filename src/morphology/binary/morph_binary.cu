#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/common/grid_block_sizes.h"
#include <stdio.h>

// Perform erosion/dilation operation for one pixel
template<typename dtype>
CUDA_HOSTDEV 
void morph_binary_pixel(dtype *image, dtype *output, 
                   int centerIdx, int centerIdy, int centerIdz, 
                   int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   const int xsize, const int ysize, const int zsize, MorphOp operation){
    dtype *im = image;
    int *ik = kernel;
    dtype aux;
    if(operation == EROSION){
        aux = 1; //erosion operation
    }
    else{//quinta comunista
        aux = 0; //dilation operation
    }
    
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
                // don't care pixels are signaled as -1 in the kernel 
                if(imageIdx < 0 || imageIdx > xsize-1 || 
                   imageIdy < 0 || imageIdy > ysize-1 || 
                   imageIdz < 0 || imageIdz > zsize-1 ||
                   ik[ix] < 0){
                    // do nothing.
                }

                else{
                    if(operation == EROSION){
                        aux = (im[index] == (dtype)ik[ix]) && aux; //erosion operation
                    } 
                    else{
                        aux = (im[index] == (dtype)ik[ix]) || aux; //dilation operation
                    }
                }

            }
            ik += kernel_xsize; 
        }
    }
    output[centerIdz*ysize*xsize + centerIdy*xsize + centerIdx] = aux;
}
template CUDA_HOSTDEV void morph_binary_pixel<int>(int *, int *, int, int, int, int *, int, int, int, const int, const int, const int, MorphOp);
template CUDA_HOSTDEV void morph_binary_pixel<u_int32_t>(u_int32_t *, u_int32_t *, int, int, int, int *, int, int, int, const int, const int, const int, MorphOp);
template CUDA_HOSTDEV void morph_binary_pixel<u_int16_t>(u_int16_t *, u_int16_t *, int, int, int, int *, int, int, int, const int, const int, const int, MorphOp);

template<typename dtype>
__global__
void morph_binary_kernel(dtype *deviceImage, dtype *deviceOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, MorphOp operation){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idz = threadIdx.z + blockIdx.z*blockDim.z;

    if (idx < xsize && idy < ysize && idz < zsize){
        morph_binary_pixel(deviceImage, deviceOutput, idx, idy, idz, 
                           kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                           xsize, ysize, zsize, operation);
    }
}
template __global__ void morph_binary_kernel<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphOp);
template __global__ void morph_binary_kernel<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphOp);
template __global__ void morph_binary_kernel<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, MorphOp);

// Slide kernel and erosion/dilation operation over all input image pixels
template<typename dtype>
void morph_binary_on_device(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, MorphOp operation, const int flag_verbose)
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
    CHECK(cudaMalloc((int**)&deviceImage, nBytes));
    CHECK(cudaMalloc((int**)&deviceOutput, nBytes));
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

    // device erosion/dialation 
    morph_binary_kernel<<<grid, block>>>(deviceImage, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                         xsize, ysize, zsize, operation);
    cudaDeviceSynchronize(); //assures all gpu threads are fineshed

    // transfer data from the device to the host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
    
    // free host memorys
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}
template void morph_binary_on_device<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphOp, const int);
template void morph_binary_on_device<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphOp, const int);
template void morph_binary_on_device<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, MorphOp, const int);

// Slide kernel and erosion/dilation operation over all input image pixels
template<typename dtype>
void morph_binary_on_host(dtype *hostImage, dtype *hostOutput, 
             int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
             const int xsize, const int ysize, const int zsize, MorphOp operation){

    for(int idz = 0; idz < zsize; idz++){
        for(int idy = 0; idy < ysize; idy++){
            for(int idx = 0; idx < xsize; idx++){

                morph_binary_pixel(hostImage, hostOutput, idx, idy, idz, 
                                     kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                                     xsize, ysize, zsize, operation);
                
            }
        }
    }// slide over image
}
template void morph_binary_on_host<int>(int *, int *, int *, int, int, int, const int, const int, const int, MorphOp);
template void morph_binary_on_host<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, MorphOp);
template void morph_binary_on_host<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, MorphOp);
