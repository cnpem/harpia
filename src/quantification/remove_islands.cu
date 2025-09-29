#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/quantification/remove_islands.h"


__global__ void label_counter_2d(int* image, int* counter, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);

    if (idx < xsize && idy < ysize)
    {
        int image_index = idy * xsize + idx;
        int counter_index = image[image_index];

        atomicAdd(&counter[counter_index],1);

    }

}

__global__ void remove_2d(int* image, int* counter, int threshold, int xsize, int ysize)
{

    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);

    if (idx < xsize && idy < ysize)
    {
        int image_index = idy * xsize + idx;
        int counter_index = image[image_index];

        if ( counter[counter_index] < threshold)
        {
            image[image_index] = 0;
        }
        

    }

}

__global__ void label_counter_3d(int* image, int* counter, int xsize, int ysize, int zsize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    if (idx < xsize && idy < ysize && idz < zsize)
    {
        int image_index = idz * xsize * ysize + idy * xsize + idx;
        int counter_index = image[image_index];

        atomicAdd(&counter[counter_index], 1);
    }
}

__global__ void remove_3d(int* image, int* counter, int threshold, int xsize, int ysize, int zsize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;

    if (idx < xsize && idy < ysize && idz < zsize)
    {
        int image_index = idz * xsize * ysize + idy * xsize + idx;
        int counter_index = image[image_index];

        if (counter[counter_index] < threshold)
        {
            image[image_index] = 0;
        }
    }
}


void remove_islands(int* image, int* output, int threshold, int xsize, int ysize, int zsize, bool type)
{

    int* dev_image;
    int* dev_output;

    cudaMalloc(&dev_image, xsize * ysize * zsize * sizeof(int));
    cudaMalloc(&dev_output, xsize * ysize * zsize* sizeof(int));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);
    
    if(type==false)
    {

	    dim3 blockDim(32, 32);
	    dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y);

	    label_counter_2d<<<gridDim,blockDim>>>(dev_image,dev_output,xsize,ysize);
	    cudaDeviceSynchronize();

	    remove_2d<<<gridDim,blockDim>>>(dev_image,dev_output,threshold,xsize,ysize);
	    
	    cudaDeviceSynchronize();
    }
    
    
    else
    {
    
        dim3 blockDim(8, 8, 8);
        dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, 
                     (ysize + blockDim.y - 1) / blockDim.y, 
                     (zsize + blockDim.z - 1) / blockDim.z);
    		
        label_counter_3d<<<gridDim,blockDim>>>(dev_image,dev_output,xsize,ysize,zsize);
	    cudaDeviceSynchronize();

	    remove_3d<<<gridDim,blockDim>>>(dev_image,dev_output,threshold,xsize,ysize,zsize);
	    
	    cudaDeviceSynchronize();
    }
	
	
    cudaMemcpy(output, dev_output, xsize * ysize * zsize *sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(image, dev_image, xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);

}

/*
int main()
{
    const int xsize = 8;
    const int ysize = 8;

    int output[xsize*ysize];
    int counter[xsize*ysize];

    int image[xsize*ysize] = 
    {
         1, 1, 0, 0, 2, 0, 0, 0,
         1, 1, 0, 0, 0, 0, 2, 1,
         1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 0,
         0, 0, 1, 1, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 0, 0, 1, 1, 0, 0
    };

    connectedComponents(image,output,xsize,ysize);

    remove_islands(output,counter,3,xsize,ysize);

    for (int i = 0; i < ysize; i++) {
        for (int j = 0; j < xsize; j++) {
            std::cout << output[i*xsize +j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/