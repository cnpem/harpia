#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/median_filter.h"

template<typename dtype>
__device__ void bubble_sort(dtype *array, int size)
{
	
    int j;
    int flag = 1;

    dtype temp;

	if (!array || size < 2)
    {
		return;
    }

	while (flag != 0)
	{
		flag = 0;

		for (j = 0; j < size - 1; j++)
        {
			if (array[j] > array[j + 1])
			{
				temp = array[j];

				array[j] = array[j + 1];

				array[j + 1] = temp;

				flag = 1;
			}

        }

	}

}

template<typename dtype>
__device__ void get_median_kernel_2d(dtype* image, dtype* kernel, int i, int j, int rows, int cols, int rows_kernel, int cols_kernel)
{

    int input_col;
    int input_row;
    
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
                kernel[(i * cols + j) * (rows_kernel * cols_kernel) + (m * cols_kernel + n)] = image[input_row * cols + input_col];
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

                kernel[(i * cols + j) * (rows_kernel * cols_kernel) + (m * cols_kernel + n)] = image[input_row * cols + input_col];

            }
            
        }

    }


}


template<typename dtype>
__device__ void get_median_kernel_3d(dtype* image, dtype* kernel,
                          int i, int j, int k, 
                          int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel)
{

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
                    kernel[(k * rows * cols + i * cols + j) * (rows_kernel * cols_kernel * depth_kernel) + (l * rows_kernel * cols_kernel) + (m * cols_kernel) + n] = image[(input_depth * rows * cols) + (input_row * cols) + input_col];
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
                    

                    kernel[(k * rows * cols + i * cols + j) * (rows_kernel * cols_kernel * depth_kernel) + (l * rows_kernel * cols_kernel) + (m * cols_kernel) + n] = image[(input_depth * rows * cols) + (input_row * cols) + input_col];

                }
                
            }

        }

    }

}



template<typename dtype>
__global__ void median_filter_kernel_2d(dtype* image, dtype* output, dtype* kernel, int rows, int cols, int idz, int rows_kernel, int cols_kernel)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        get_median_kernel_2d(image + idz * rows * cols, kernel, idx, idy, rows, cols, rows_kernel, cols_kernel);
        bubble_sort(kernel + (idx * cols + idy) * (rows_kernel * cols_kernel), rows_kernel * cols_kernel);

        int median_index = (rows_kernel * cols_kernel) / 2;

        if ((rows_kernel * cols_kernel) % 2 == 0)
        {
            dtype median_value = (kernel[(idx * cols + idy) * (rows_kernel * cols_kernel) + median_index] +
                                  kernel[(idx * cols + idy) * (rows_kernel * cols_kernel) + median_index - 1]) / 2;
            output[idz * rows * cols + idx * cols + idy] = median_value;
        }

        else
        {
            output[idz * rows * cols + idx * cols + idy] = kernel[(idx * cols + idy) * (rows_kernel * cols_kernel) + median_index];
        }

    }

}

template<typename dtype>
__global__ void median_filter_kernel_3d(dtype* image, dtype* output, dtype* kernel,
                                        int rows, int cols, int depth, int idz,
                                        int rows_kernel, int cols_kernel, int depth_kernel)
{

    //threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        get_median_kernel_3d(image, kernel, idx, idy, idz, rows, cols, depth, rows_kernel, cols_kernel, depth_kernel);
        bubble_sort(kernel + (idz * rows * cols + idx * cols + idy) * (rows_kernel * cols_kernel * depth_kernel), rows_kernel * cols_kernel * depth_kernel);

        int median_index = (rows_kernel * cols_kernel * depth_kernel) / 2;

        if ((rows_kernel * cols_kernel * depth_kernel) % 2 == 0)
        {
            dtype median_value = (kernel[(idz * rows * cols + idx * cols + idy) * (rows_kernel * cols_kernel*depth_kernel) + median_index] +
                                kernel[(idz * rows * cols + idx * cols + idy) * (rows_kernel * cols_kernel*depth_kernel) + median_index - 1]) / 2;
            output[idz * rows * cols + idx * cols + idy] = median_value;
        }
        else
        {
            output[idz * rows * cols + idx * cols + idy] = kernel[(idz * rows * cols + idx * cols + idy) * (rows_kernel * cols_kernel * depth_kernel) + median_index];
        }

    }   

    
}


template __global__ void median_filter_kernel_2d<int>(int* image, int* output, int* kernel, int rows, int cols, int idz, int rows_kernel, int cols_kernel);
template __global__ void median_filter_kernel_2d<float>(float* image, float* output, float* kernel, int rows, int cols, int idz, int rows_kernel, int cols_kernel);

template __global__ void median_filter_kernel_3d<int>(int* image, int* output, int* kernel,
                                                      int rows, int cols, int depth, int idz,
                                                      int rows_kernel, int cols_kernel, int depth_kernel);

template __global__ void median_filter_kernel_3d<float>(float* image, float* output, float* kernel,
                                                        int rows, int cols, int depth, int idz,
                                                        int rows_kernel, int cols_kernel, int depth_kernel);




template<typename dtype>
void median_filtering(dtype* image, dtype* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel)
{

    dtype* dev_image;
    dtype* dev_output;
    dtype* dev_kernel;

    cudaMalloc((void**)&dev_image, rows * cols * depth * sizeof(dtype));
    cudaMalloc((void**)&dev_output, rows * cols * depth * sizeof(dtype));
    cudaMalloc((void**)&dev_kernel, rows * cols * rows_kernel * cols_kernel * depth_kernel * sizeof(dtype));

    cudaMemcpy(dev_image, image, rows * cols * depth * sizeof(dtype), cudaMemcpyHostToDevice);

    if (depth_kernel == 1)
    {

        dim3 blockSize(32, 32);
        dim3 gridSize((rows + blockSize.y - 1) / blockSize.y,  (cols + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < depth; ++k)
        {
            median_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, rows, cols, k, rows_kernel, cols_kernel);

            cudaDeviceSynchronize();
        }
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  
    }

    else
    {

        dim3 blockSize(32, 32);
        dim3 gridSize((rows + blockSize.y - 1) / blockSize.y,  (cols + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < depth; ++k)
        {
            median_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, rows, cols, depth, k, rows_kernel, cols_kernel, depth_kernel);

            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

    }

    cudaMemcpy(output, dev_output, rows * cols * depth * sizeof(dtype), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
    cudaFree(dev_kernel);
}

// Explicit instantiation for dtype
template void median_filtering<float>(float* image, float* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);
template void median_filtering<int>(int* image, int* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);
template void median_filtering<unsigned int>(unsigned int* image, unsigned int* output, int rows, int cols, int depth, int rows_kernel, int cols_kernel, int depth_kernel);