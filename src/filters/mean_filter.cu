#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/mean_filter.h"


template<typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int xsize, int ysize, int idz, int kx, int ky)
{

    //threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        //mean value
        float mean = 0;

        //get the neighbors
        get_mean_kernel_2d(image + idz * xsize * ysize, &mean, idx, idy, xsize, ysize, kx, ky);

        //assign the mean value
        output[idz * xsize * ysize + idx * ysize + idy] = mean;
    }   
    
}

template<typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output,
                                    int xsize, int ysize, int zsize,
                                    int kx, int ky, int kz)
{

    //threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < xsize && idy < ysize && idz < zsize)
    {
        
        //mean value
        float mean = 0;

        //get the neighbors
        get_mean_kernel_3d(image,&mean,
                        idx,idy,idz,
                        xsize,ysize,zsize,
                        kx,ky, kz);


        //assign the mean value
        output[idz * ysize * xsize + idx * ysize + idy] = mean;

    }

    
}


template __global__ void mean_filter_kernel_2d<int>(int* image, float* output, int xsize, int ysize, int idz, int kx, int ky);
template __global__ void mean_filter_kernel_2d<float>(float* image, float* output, int xsize, int ysize, int idz, int kx, int ky);

template __global__ void mean_filter_kernel_3d<int>(int* image, float* output,int xsize, int ysize, int zsize,int kx, int ky, int kz);
template __global__ void mean_filter_kernel_3d<float>(float* image, float* output,int xsize, int ysize, int zsize,int kx, int ky, int kz);




template<typename dtype>
void mean_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz)
{

    dtype* dev_image;
    float* dev_output;

    cudaMalloc((void**)&dev_image, xsize * ysize * zsize * sizeof(dtype));
    cudaMalloc((void**)&dev_output, xsize * ysize * zsize * sizeof(float));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

    if (kz == 1)
    {

        dim3 blockSize(32, 32);
        dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y,  (ysize + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < zsize; ++k)
        {
            mean_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, xsize, ysize, k, kx, ky);

            cudaDeviceSynchronize();
        }
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  
    }

    else
    {

        dim3 blockSize(8, 8, 8);
        dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y,  (ysize + blockSize.x - 1) / blockSize.x, (zsize + blockSize.z - 1) / blockSize.z);

        auto start = std::chrono::high_resolution_clock::now();

        mean_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, xsize, ysize, zsize, kx, ky, kz);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

    }

    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
}

// Explicit instantiation for float
template void mean_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);
template void mean_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);
template void mean_filtering<unsigned int>(unsigned int* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);
