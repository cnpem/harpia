#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"mean_filter.h"

template<typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int rows, int cols, int rows_kernel, int cols_kernel)
{

    int input_col;
    int input_row;

    float accumulation = 0;
    
    for(int m = 0; m < rows_kernel; m++)
    {

        for (int n = 0; n < cols_kernel; n++)
        {
            //this is needed to compute everything with respect to the center of the kernel.
            input_row = i - rows_kernel / 2 + m;
            input_col = j - cols_kernel / 2 + n;

            // Check if input_row and input_col are within bounds
            if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols)
            {
                accumulation += image[input_row * cols + input_col];
            }
            
            //make a padding function to substitute this line of code.
            else
            {

                // Reflect padding
                if (input_row < 0)
                    input_row = -input_row;
                else if (input_row >= rows)
                    input_row = 2 * rows - input_row - 1;

                if (input_col < 0)
                    input_col = -input_col;
                else if (input_col >= cols)
                    input_col = 2 * cols - input_col - 1;

                accumulation += image[input_row * cols + input_col];

            }
            
        }

    }

    *mean = accumulation/(rows_kernel*cols_kernel);

}

template<typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean,
                          int i, int j, int k, 
                          int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel)
{

    float accumulation = 0;

    int input_col;
    int input_row;
    int input_depth;
    
    for (int l = 0; l < depth_kernel; l++)
    {

        for(int m = 0; m < rows_kernel; m++)
        {

            for (int n = 0; n < cols_kernel; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                input_row = i - rows_kernel / 2 + m;
                input_col = j - cols_kernel / 2 + n;
                input_depth = k - depth_kernel / 2 + l;

                if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols && input_depth >= 0 && input_depth < depth)
                {
                    accumulation += image[(input_depth * rows * cols) + (input_row * cols) + input_col];
                }
                
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (input_row < 0)
                    {
                        input_row = -input_row;
                    }

                    else if (input_row >= rows)
                    {
                        input_row = 2 * rows - input_row - 1;
                    }


                    if (input_col < 0)
                    {
                        input_col = -input_col;
                    }

                    else if (input_col >= cols)
                    {
                        input_col = 2 * cols - input_col - 1;
                    }

                    if (input_depth < 0)
                    {
                        input_depth = - input_depth;
                    }

                    else if (input_depth>=depth)
                    {
                        input_depth = 2 * depth - input_depth -1;
                    }
                    

                    accumulation += image[(input_depth * rows * cols) + (input_row * cols) + input_col];

                }
                
            }

        }

    }

    *mean = accumulation/(rows_kernel*cols_kernel*depth_kernel);

}


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