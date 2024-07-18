#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/mean_filter.h"


template<typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int rows, int cols, int idz, int rows_kernel, int cols_kernel)
{

    //threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        //mean value
        float mean = 0;

        //get the neighbors
        get_mean_kernel_2d(image + idz * rows * cols, &mean, idx, idy, rows, cols, rows_kernel, cols_kernel);

        //assign the mean value
        output[idz * rows * cols + idx * cols + idy] = mean;
    }   
    
}

template<typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)
{

    //threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < rows && idy < cols && idz < depth)
    {
        
        //mean value
        float mean = 0;

        //get the neighbors
        get_mean_kernel_3d(image,&mean,
                        idx,idy,idz,
                        rows,cols,depth,
                        rows_kernel,cols_kernel, depth_kernel);


        //assign the mean value
        output[idz * cols * rows + idx * cols + idy] = mean;

    }

    
}


template __global__ void mean_filter_kernel_2d<int>(int* image, float* output, int rows, int cols, int idz, int rows_kernel, int cols_kernel);
template __global__ void mean_filter_kernel_2d<float>(float* image, float* output, int rows, int cols, int idz, int rows_kernel, int cols_kernel);

template __global__ void mean_filter_kernel_3d<int>(int* image, float* output,int rows, int cols, int depth,int rows_kernel, int cols_kernel, int depth_kernel);
template __global__ void mean_filter_kernel_3d<float>(float* image, float* output,int rows, int cols, int depth,int rows_kernel, int cols_kernel, int depth_kernel);




template<typename dtype>
void mean_filtering(dtype* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel)
{

    dtype* dev_image;
    float* dev_output;

    cudaMalloc((void**)&dev_image, rows * cols * depth * sizeof(dtype));
    cudaMalloc((void**)&dev_output, rows * cols * depth * sizeof(float));

    cudaMemcpy(dev_image, image, rows * cols * depth * sizeof(dtype), cudaMemcpyHostToDevice);

    if (depth_kernel == 1)
    {

        dim3 blockSize(32, 32);
        dim3 gridSize((rows + blockSize.y - 1) / blockSize.y,  (cols + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < depth; ++k)
        {
            mean_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, rows, cols, k, rows_kernel, cols_kernel);

            cudaDeviceSynchronize();
        }
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  
    }

    else
    {

        dim3 blockSize(8, 8, 8);
        dim3 gridSize((rows + blockSize.y - 1) / blockSize.y,  (cols + blockSize.x - 1) / blockSize.x, (depth + blockSize.z - 1) / blockSize.z);

        auto start = std::chrono::high_resolution_clock::now();

        mean_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, rows, cols, depth, rows_kernel, cols_kernel, depth_kernel);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

    }

    cudaMemcpy(output, dev_output, rows * cols * depth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
}

// Explicit instantiation for float
template void mean_filtering<float>(float* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);
template void mean_filtering<int>(int* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);
template void mean_filtering<unsigned int>(unsigned int* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);
