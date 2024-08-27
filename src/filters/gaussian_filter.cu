#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/gaussian_filter.h"

template <typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz,
                                          int xsize, int ysize, int zsize, int nx, int ny) {

  //threads indices
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  // general matrix convolution for each pixel of the image.
  if (idx < xsize && idy < ysize) {
    //temp variable
    float temp;

    //convolution.
    convolution2d(image + idz * xsize * ysize, &temp, deviceKernel, idx, idy, xsize, ysize, nx, ny);

    output[idz * xsize * ysize + idx * ysize + idy] = (float)temp;
  }
}

template <typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, float* deviceKernel,
                                          int xsize, int ysize, int zsize, int nx, int ny, int nz) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    float temp;

    convolution3d(image, &temp, deviceKernel, idx, idy, idz, xsize, ysize, zsize, nx, ny, nz);

    output[idz * xsize * ysize + idx * ysize + idy] = (float)temp;
  }
}

template __global__ void gaussian_filter_kernel_2d<int>(int* image, float* output,
                                                        float* deviceKernel, int idz, int xsize,
                                                        int ysize, int zsize, int nx, int ny);
template __global__ void gaussian_filter_kernel_2d<float>(float* image, float* output,
                                                          float* deviceKernel, int idz, int xsize,
                                                          int ysize, int zsize, int nx, int ny);

template __global__ void gaussian_filter_kernel_3d<int>(int* image, float* output,
                                                        float* deviceKernel, int xsize, int ysize,
                                                        int zsize, int nx, int ny, int nz);
template __global__ void gaussian_filter_kernel_3d<float>(float* image, float* output,
                                                          float* deviceKernel, int xsize, int ysize,
                                                          int zsize, int nx, int ny, int nz);

template <typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma,
                        bool type) {

  dtype* deviceImage;
  float* deviceOutput;
  cudaMalloc((void**)&deviceImage, xsize * ysize * zsize * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, xsize * ysize * zsize * sizeof(float));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    //kernel size
    int nx = (int)ceil(2 * sigma + 1);
    int ny = nx;

    float* kernel;
    get_gaussian_kernel_2d(&kernel, nx, ny, sigma);

    float* deviceKernel;
    cudaMalloc((void**)&deviceKernel, nx * ny * sizeof(float));
    cudaMemcpy(deviceKernel, kernel, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel, k,
                                                         xsize, ysize, zsize, nx, ny);

      cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernel);
  }

  else {
    //kernel size
    int nx = (int)ceil(2 * sigma + 1);
    int ny = nx;
    int nz = nx;

    float* kernel;
    get_gaussian_kernel_3d(&kernel, nx, ny, nz, sigma);

    float* deviceKernel;
    cudaMalloc((void**)&deviceKernel, nx * ny * nz * sizeof(float));
    cudaMemcpy(deviceKernel, kernel, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x,
                  (zsize + blockSize.z - 1) / blockSize.z);

    auto start = std::chrono::high_resolution_clock::now();

    gaussian_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel,
                                                       xsize, ysize, zsize, nx, ny, nz);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernel);
  }

  cudaMemcpy(output, deviceOutput, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

// Explicit instantiation
template void gaussian_filtering<float>(float* image, float* output, int xsize, int ysize,
                                        int zsize, float sigma, bool type);
template void gaussian_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize,
                                      float sigma, bool type);
template void gaussian_filtering<unsigned int>(unsigned int* image, float* output, int xsize,
                                               int ysize, int zsize, float sigma, bool type);

/*
int main()
{
    int xsize = 512;
    int ysize = 512;
    int zsize = 512;

    static float* image;
    image = (float*)malloc(zsize*xsize*ysize*sizeof(int));

    static float* output;
    output = (float*)malloc(zsize*xsize*ysize*sizeof(int));

    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                if (i!=j)
                {
                    image[k * xsize * ysize + i * ysize + j] = 1;
                }

                if (i==j)
                {
                    image[k * xsize * ysize + i * ysize + j] = i+j;
                }
                
        
                output[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }

    float sigma = 6.;
    gaussian_filtering(image,output,xsize,ysize,zsize,sigma,true);
    
    
    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                std::cout<<output[k*xsize*ysize + i*ysize +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    

    

    return 0;
}
*/