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
__device__ void get_median_kernel_2d(dtype* image, dtype* kernel, int i, int j, int xsize, int ysize, int kx, int ky)
{

    int input_col;
    int input_row;
    
    for(int m = 0; m < kx; m++)
    {

        for (int n = 0; n < ky; n++)
        {
            //this is needed to compute everything with respect to the center of the kernel.
            input_row = i - kx / 2 + m;
            input_col = j - ky / 2 + n;

            // Check if input_row and input_col are within bounds
            if (input_row >= 0 && input_row < xsize && input_col >= 0 && input_col < ysize)
            {
                kernel[(i * ysize + j) * (kx * ky) + (m * ky + n)] = image[input_row * ysize + input_col];
            }
            
            //make a padding function to substitute this line of code.
            else
            {

                // Reflect padding
                if (input_row < 0)
                    input_row = -input_row;
                else if (input_row >= xsize)
                    input_row = 2 * xsize - input_row - 1;

                if (input_col < 0)
                    input_col = -input_col;
                else if (input_col >= ysize)
                    input_col = 2 * ysize - input_col - 1;

                kernel[(i * ysize + j) * (kx * ky) + (m * ky + n)] = image[input_row * ysize + input_col];

            }
            
        }

    }


}


template<typename dtype>
__device__ void get_median_kernel_3d(dtype* image, dtype* kernel,
                          int i, int j, int k, 
                          int xsize, int ysize, int zsize,
                          int kx, int ky, int kz)
{

    int input_col;
    int input_row;
    int input_zsize;
    
    for (int l = 0; l < kz; l++)
    {

        for(int m = 0; m < kx; m++)
        {

            for (int n = 0; n < ky; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                input_row = i - kx / 2 + m;
                input_col = j - ky / 2 + n;
                input_zsize = k - kz / 2 + l;

                if (input_row >= 0 && input_row < xsize && input_col >= 0 && input_col < ysize && input_zsize >= 0 && input_zsize < zsize)
                {
                    kernel[(k * xsize * ysize + i * ysize + j) * (kx * ky * kz) + (l * kx * ky) + (m * ky) + n] = image[(input_zsize * xsize * ysize) + (input_row * ysize) + input_col];
                }
                
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (input_row < 0)
                    {
                        input_row = -input_row;
                    }

                    else if (input_row >= xsize)
                    {
                        input_row = 2 * xsize - input_row - 1;
                    }


                    if (input_col < 0)
                    {
                        input_col = -input_col;
                    }

                    else if (input_col >= ysize)
                    {
                        input_col = 2 * ysize - input_col - 1;
                    }

                    if (input_zsize < 0)
                    {
                        input_zsize = - input_zsize;
                    }

                    else if (input_zsize>=zsize)
                    {
                        input_zsize = 2 * zsize - input_zsize -1;
                    }
                    

                    kernel[(k * xsize * ysize + i * ysize + j) * (kx * ky * kz) + (l * kx * ky) + (m * ky) + n] = image[(input_zsize * xsize * ysize) + (input_row * ysize) + input_col];

                }
                
            }

        }

    }

}



template<typename dtype>
__global__ void median_filter_kernel_2d(dtype* image, dtype* output, dtype* kernel, int xsize, int ysize, int idz, int kx, int ky)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        get_median_kernel_2d(image + idz * xsize * ysize, kernel, idx, idy, xsize, ysize, kx, ky);
        bubble_sort(kernel + (idx * ysize + idy) * (kx * ky), kx * ky);

        int median_index = (kx * ky) / 2;

        if ((kx * ky) % 2 == 0)
        {
            dtype median_value = (kernel[(idx * ysize + idy) * (kx * ky) + median_index] +
                                  kernel[(idx * ysize + idy) * (kx * ky) + median_index - 1]) / 2;
            output[idz * xsize * ysize + idx * ysize + idy] = median_value;
        }

        else
        {
            output[idz * xsize * ysize + idx * ysize + idy] = kernel[(idx * ysize + idy) * (kx * ky) + median_index];
        }

    }

}

template<typename dtype>
__global__ void median_filter_kernel_3d(dtype* image, dtype* output, dtype* kernel,
                                        int xsize, int ysize, int zsize, int idz,
                                        int kx, int ky, int kz)
{

    //threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        get_median_kernel_3d(image, kernel, idx, idy, idz, xsize, ysize, zsize, kx, ky, kz);
        bubble_sort(kernel + (idz * xsize * ysize + idx * ysize + idy) * (kx * ky * kz), kx * ky * kz);

        int median_index = (kx * ky * kz) / 2;

        if ((kx * ky * kz) % 2 == 0)
        {
            dtype median_value = (kernel[(idz * xsize * ysize + idx * ysize + idy) * (kx * ky*kz) + median_index] +
                                kernel[(idz * xsize * ysize + idx * ysize + idy) * (kx * ky*kz) + median_index - 1]) / 2;
            output[idz * xsize * ysize + idx * ysize + idy] = median_value;
        }
        else
        {
            output[idz * xsize * ysize + idx * ysize + idy] = kernel[(idz * xsize * ysize + idx * ysize + idy) * (kx * ky * kz) + median_index];
        }

    }   

    
}


template __global__ void median_filter_kernel_2d<int>(int* image, int* output, int* kernel, int xsize, int ysize, int idz, int kx, int ky);
template __global__ void median_filter_kernel_2d<float>(float* image, float* output, float* kernel, int xsize, int ysize, int idz, int kx, int ky);

template __global__ void median_filter_kernel_3d<int>(int* image, int* output, int* kernel,
                                                      int xsize, int ysize, int zsize, int idz,
                                                      int kx, int ky, int kz);

template __global__ void median_filter_kernel_3d<float>(float* image, float* output, float* kernel,
                                                        int xsize, int ysize, int zsize, int idz,
                                                        int kx, int ky, int kz);




template<typename dtype>
void median_filtering(dtype* image, dtype* output, int xsize, int ysize, int zsize, int kx, int ky, int kz)
{

    dtype* dev_image;
    dtype* dev_output;
    dtype* dev_kernel;

    cudaMalloc((void**)&dev_image, xsize * ysize * zsize * sizeof(dtype));
    cudaMalloc((void**)&dev_output, xsize * ysize * zsize * sizeof(dtype));
    cudaMalloc((void**)&dev_kernel, xsize * ysize * kx * ky * kz * sizeof(dtype));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

    if (kz == 1)
    {

        dim3 blockSize(32, 32);
        dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y,  (ysize + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < zsize; ++k)
        {
            median_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, xsize, ysize, k, kx, ky);

            cudaDeviceSynchronize();
        }
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  
    }

    else
    {

        dim3 blockSize(32, 32);
        dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y,  (ysize + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < zsize; ++k)
        {
            median_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, xsize, ysize, zsize, k, kx, ky, kz);

            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

    }

    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
    cudaFree(dev_kernel);
}

// Explicit instantiation for dtype
template void median_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);
template void median_filtering<int>(int* image, int* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);
template void median_filtering<unsigned int>(unsigned int* image, unsigned int* output, int xsize, int ysize, int zsize, int kx, int ky, int kz);