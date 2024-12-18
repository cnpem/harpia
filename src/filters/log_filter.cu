#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/log_filter.h"

void get_laplacian_kernel_2d(float** kernel) {
  /*

        Laplacian hostKernel has the form:

          +-----------------+
          |   -1  -1  -1    |
          |   -1   8  -1    |
          |   -1  -1  -1    |
          +-----------------+

    */

  *kernel = (float*)malloc(sizeof(float) * 9);

  if (!*kernel) {
    return;
  }

  (*kernel)[0] = -1;
  (*kernel)[1] = -1;
  (*kernel)[2] = -1;

  (*kernel)[3] = -1;
  (*kernel)[4] = 8;
  (*kernel)[5] = -1;

  (*kernel)[6] = -1;
  (*kernel)[7] = -1;
  (*kernel)[8] = -1;
}

void get_laplacian_kernel_3d(float** kernel) {
  /*

        Laplacian hostKernel has the form:

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

  *kernel = (float*)malloc(sizeof(float) * 27);

  if (!*kernel) {
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

template <typename dtype>
__global__ void log_filter_kernel_2d(dtype* image, float* output, float* deviceKernel, int idz,
                                     int xsize, int ysize, int zsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    float temp;

    convolution2d(image + idz * xsize * ysize, &temp, deviceKernel, idx, idy, xsize, ysize, 3, 3);

    output[idz * xsize * ysize + idx * ysize + idy] = (float)sqrtf(temp * temp);
  }
}

template <typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* deviceKernel, int xsize,
                                     int ysize, int zsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  //change xsize and ysize notation-->you made a mistake dummy.
  if (idx < xsize && idy < ysize && idz < zsize) {
    float temp;

    convolution3d(image, &temp, deviceKernel, idx, idy, idz, xsize, ysize, zsize, 3, 3, 3);

    output[idz * xsize * ysize + idx * ysize + idy] = (float)sqrtf(temp * temp);
  }
}

template __global__ void log_filter_kernel_2d<int>(int* image, float* output, float* deviceKernel,
                                                   int idz, int xsize, int ysize, int zsize);
template __global__ void log_filter_kernel_2d<float>(float* image, float* output,
                                                     float* deviceKernel, int idz, int xsize,
                                                     int ysize, int zsize);

template __global__ void log_filter_kernel_3d<int>(int* image, float* output, float* deviceKernel,
                                                   int xsize, int ysize, int zsize);
template __global__ void log_filter_kernel_3d<float>(float* image, float* output,
                                                     float* deviceKernel, int xsize, int ysize,
                                                     int zsize);

template <typename dtype>
void log_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type) {

  dtype* deviceImage;
  float* deviceOutput;
  cudaMalloc((void**)&deviceImage, xsize * ysize * zsize * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, xsize * ysize * zsize * sizeof(float));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    float* kernel;
    get_laplacian_kernel_2d(&kernel);

    float* deviceKernel;
    cudaMalloc((void**)&deviceKernel, 9 * sizeof(float));
    cudaMemcpy(deviceKernel, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      log_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel, k,
                                                    xsize, ysize, zsize);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernel);
  }

  else {
    float* kernel;
    get_laplacian_kernel_3d(&kernel);

    float* deviceKernel;
    cudaMalloc((void**)&deviceKernel, 27 * sizeof(float));
    cudaMemcpy(deviceKernel, kernel, 27 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x,
                  (zsize + blockSize.z - 1) / blockSize.z);

    auto start = std::chrono::high_resolution_clock::now();

    log_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel, xsize,
                                                  ysize, zsize);

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
template void log_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize,
                                   bool type);
template void log_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize,
                                 bool type);
template void log_filtering<unsigned int>(unsigned int* image, float* output, int xsize, int ysize,
                                          int zsize, bool type);
