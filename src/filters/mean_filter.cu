#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/mean_filter.h"

template <typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int xsize, int ysize, int idz,
                                      int nx, int ny) {

  //threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    //mean value
    float mean = 0;

    //get the neighbors
    get_mean_kernel_2d(image + idz * xsize * ysize, &mean, idx, idy, xsize, ysize, nx, ny);

    //assign the mean value
    output[idz * xsize * ysize + idx * ysize + idy] = mean;
  }
}

template <typename dtype>
__global__ void mean_filter_kernel_3d(dtype* image, float* output, int xsize, int ysize, int zsize,
                                      int nx, int ny, int nz) {

  //threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < xsize && idy < ysize && idz < zsize) {

    //mean value
    float mean = 0;

    //get the neighbors
    get_mean_kernel_3d(image, &mean, idx, idy, idz, xsize, ysize, zsize, nx, ny, nz);

    //assign the mean value
    output[idz * ysize * xsize + idx * ysize + idy] = mean;
  }
}

template __global__ void mean_filter_kernel_2d<int>(int* image, float* output, int xsize, int ysize,
                                                    int idz, int nx, int ny);
template __global__ void mean_filter_kernel_2d<float>(float* image, float* output, int xsize,
                                                      int ysize, int idz, int nx, int ny);

template __global__ void mean_filter_kernel_3d<int>(int* image, float* output, int xsize, int ysize,
                                                    int zsize, int nx, int ny, int nz);
template __global__ void mean_filter_kernel_3d<float>(float* image, float* output, int xsize,
                                                      int ysize, int zsize, int nx, int ny, int nz);

template <typename dtype>
void mean_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, int nx, int ny,
                    int nz) {

  dtype* deviceImage;
  float* deviceOutput;

  cudaMalloc((void**)&deviceImage, xsize * ysize * zsize * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, xsize * ysize * zsize * sizeof(float));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

  if (nz == 1) {

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      mean_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, xsize, ysize, k, nx,
                                                     ny);

      cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  }

  else {

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x,
                  (zsize + blockSize.z - 1) / blockSize.z);

    auto start = std::chrono::high_resolution_clock::now();

    mean_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, xsize, ysize, zsize,
                                                   nx, ny, nz);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  }

  cudaMemcpy(output, deviceOutput, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

// Explicit instantiation for float
template void mean_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize,
                                    int nx, int ny, int nz);
template void mean_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize,
                                  int nx, int ny, int nz);
template void mean_filtering<unsigned int>(unsigned int* image, float* output, int xsize, int ysize,
                                           int zsize, int nx, int ny, int nz);
