#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/quantification/fraction.h"

__global__ void fraction_counter(int* image, int* counter, int acumulator, int xsize, int ysize, int zsize)
{


    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);
    int idz = (threadIdx.z + blockIdx.z * blockDim.z);

    if (idx < xsize && idy < ysize && idz < zsize)
    {
        int image_index = idz * xsize * ysize + idy * xsize + idx;
        int counter_index = image[image_index];

        atomicAdd(&counter[counter_index],1);

        atomicAdd(&acumulator,1);

    }

}

__global__ void labels_fraction(int* image, int* counter, int acumulator, int xsize, int ysize, int zsize)
{

    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);
    int idz = (threadIdx.z + blockIdx.z * blockDim.z);

    if (idx < xsize && idy < ysize && idz < zsize)
    {
        int index = idz * xsize * ysize + idy * xsize + idx;

        counter[index] = 100*counter[index] / acumulator ;

    }

}

void fraction(int* image, int* output, int xsize, int ysize, int zsize)
{

    int* dev_image;
    int* dev_output;
    int* dev_acumulator;

    cudaMalloc(&dev_image, xsize * ysize * zsize * sizeof(int));
    cudaMalloc(&dev_output, xsize * ysize * zsize * sizeof(int));
    cudaMalloc(&dev_acumulator, 1 * sizeof(int));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8, 8);  // Example block dimensions, can be adjusted
    dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x,
                 (ysize + blockDim.y - 1) / blockDim.y,
                 (zsize + blockDim.z - 1) / blockDim.z);

    fraction_counter<<<gridDim,blockDim>>>(dev_image,dev_output,*dev_acumulator,xsize,ysize,zsize);

    labels_fraction<<<gridDim,blockDim>>>(dev_image,dev_output,*dev_acumulator,xsize,ysize,zsize);

    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(image, dev_image, xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);

}
