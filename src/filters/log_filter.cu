#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/log_filter.h"

void get_laplacian_kernel_2d(float** kernel)
{
    /*

        Laplacian kernel has the form:

          +-----------------+
          |   -1  -1  -1    |
          |   -1   8  -1    |
          |   -1  -1  -1    |
          +-----------------+

    */

    *kernel = (float*)malloc(sizeof(float)*9);

    if (! *kernel)
    {
        return;
    }
    

    (*kernel)[0] = -1;
    (*kernel)[1] = -1;
    (*kernel)[2] = -1;

    (*kernel)[3] = -1;
    (*kernel)[4] =  8;
    (*kernel)[5] = -1;

    (*kernel)[6] = -1;
    (*kernel)[7] = -1;
    (*kernel)[8] = -1;
    

}

void get_laplacian_kernel_3d(float** kernel)
{
    /*

        Laplacian kernel has the form:

                  +--------------+
                 /     0 0 0    /|
                /      0 1 0   / |
               /       0 0 0  /  |
              +--------------+   |
             /  0  1  0     /|  /
            /   1 -6  1    / | /
           /    0  1  0   /  |/
          +--------------+   +
          |   0  0  0    |  /
          |   0  1  0    | /
          |   0  0  0    |/
          +--------------+


    */

    *kernel = (float*)malloc(sizeof(float)*27);

    if (! *kernel)
    {
        return;
    }
    
    //first plane
    (*kernel)[0] = 0;
    (*kernel)[1] = 0;
    (*kernel)[2] = 0;

    (*kernel)[3] = 0;
    (*kernel)[4] = 1;
    (*kernel)[5] = 0;

    (*kernel)[6] = 0;
    (*kernel)[7] = 0;
    (*kernel)[8] = 0;

    //second plane
    (*kernel)[9] = 0;
    (*kernel)[10] = 1;
    (*kernel)[11] = 0;

    (*kernel)[12] = 1;
    (*kernel)[13] = -6;
    (*kernel)[14] = 1;

    (*kernel)[15] = 0;
    (*kernel)[16] = 1;
    (*kernel)[17] = 0;

    //third plane
    (*kernel)[18] = 0;
    (*kernel)[19] = 0;
    (*kernel)[20] = 0;

    (*kernel)[21] = 0;
    (*kernel)[22] = 1;
    (*kernel)[23] = 0;

    (*kernel)[24] = 0;
    (*kernel)[25] = 0;
    (*kernel)[26] = 0;
}

template<typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* dev_kernel, int idz, int rows, int cols, int slices)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        float temp;

        convolution_2d(image + idz * rows * cols, &temp, dev_kernel, idx, idy, rows, cols, 3, 3);

        output[idz * rows * cols + idx * cols + idy] = (float)sqrtf(temp * temp);
    }
}

template<typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* dev_kernel, int rows, int cols, int depth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    //change rows and cols notation-->you made a mistake dummy.
    if (idx < rows && idy < cols && idz < depth)
    {
        float temp;

        convolution_3d(image, &temp, dev_kernel, idx, idy, idz, rows, cols, depth, 3, 3, 3);

        output[idz * rows * cols + idx * cols + idy] = (float)sqrtf(temp*temp);
    }
}

template __global__ void log_filter_kernel_2d<int>(int* image, float* output, float* dev_kernel, int idz,int rows, int cols,int slices);
template __global__ void log_filter_kernel_2d<float>(float* image, float* output, float* dev_kernel, int idz,int rows, int cols,int slices);

template __global__ void log_filter_kernel_3d<int>(int* image, float* output, float* dev_kernel, int rows, int cols, int depth);
template __global__ void log_filter_kernel_3d<float>(float* image, float* output, float* dev_kernel, int rows, int cols, int depth);



template<typename dtype>
void log_filtering(dtype* image, float* output, int rows, int cols, int slices, bool type)
{

    dtype* dev_image;
    float* dev_output;
    cudaMalloc((void**)&dev_image, rows * cols * slices * sizeof(dtype));
    cudaMalloc((void**)&dev_output, rows * cols * slices * sizeof(float));

    cudaMemcpy(dev_image, image, rows * cols * slices * sizeof(dtype), cudaMemcpyHostToDevice);

    if (type == false)
    {
        float* kernel;
        get_laplacian_kernel_2d(&kernel);

        float* dev_kernel;
        cudaMalloc((void**)&dev_kernel, 9 * sizeof(float));
        cudaMemcpy(dev_kernel, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(32, 32);
        dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < slices; ++k)
        {
            log_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, k, rows, cols, slices);
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

        cudaFree(dev_kernel);   
    }

    else
    {
        float* kernel;
        get_laplacian_kernel_3d(&kernel);

        float* dev_kernel;
        cudaMalloc((void**)&dev_kernel, 27 * sizeof(float));
        cudaMemcpy(dev_kernel, kernel, 27 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(8, 8, 8);
        dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x, (slices + blockSize.z - 1) / blockSize.z);

        auto start = std::chrono::high_resolution_clock::now();

        log_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output,dev_kernel,rows, cols, slices);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

        cudaFree(dev_kernel);
    }

    cudaMemcpy(output, dev_output, rows * cols * slices * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
}

// Explicit instantiation
template void log_filtering<float>(float* image, float* output, int rows, int cols, int slices, bool type);
template void log_filtering<int>(int* image, float* output, int rows, int cols, int slices, bool type);
template void log_filtering<unsigned int>(unsigned int* image, float* output, int rows, int cols, int slices, bool type);

