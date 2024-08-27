#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/threshold/adaptative_mean.h"

template <typename dtype>
__global__ void local_mean_kernel_2d(dtype* image, float* output, float weight, int rows, int cols,
                                     int idz, int rows_kernel, int cols_kernel) {

  //threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < rows && idy < cols) {
    //mean value
    float mean = 0;

    //get the mean value
    get_mean_kernel_2d(image + idz * rows * cols, &mean, idx, idy, rows, cols, rows_kernel,
                       cols_kernel);

    //apply local_mean threshold: T_{local_mean} (i,j) = mean(i,j) - w * std(i,j)
    //threshold value
    float T_local_mean = mean - weight;

    if (image[idz * rows * cols + idx * cols + idy] > T_local_mean) {
      output[idz * rows * cols + idx * cols + idy] = 255;

      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template <typename dtype>
__global__ void local_mean_kernel_3d(dtype* image, float* output, float weight, int rows, int cols,
                                     int depth, int rows_kernel, int cols_kernel,
                                     int depth_kernel) {

  //threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < rows && idy < cols && idz < depth) {

    //mean value
    float mean = 0;

    //get the mean value
    get_mean_kernel_3d(image, &mean, idx, idy, idz, rows, cols, depth, rows_kernel, cols_kernel,
                       depth_kernel);

    //apply local_mean threshold: T_{local_mean} (i,j,k) = mean(i,j,k) - w * std(i,j,k)
    //threshold value
    float T_local_mean = mean - weight;

    if (image[idz * rows * cols + idx * cols + idy] > T_local_mean) {
      output[idz * rows * cols + idx * cols + idy] = 255;

      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template __global__ void local_mean_kernel_2d<int>(int* image, float* output, float weight,
                                                   int rows, int cols, int idz, int rows_kernel,
                                                   int cols_kernel);
template __global__ void local_mean_kernel_2d<float>(float* image, float* output, float weight,
                                                     int rows, int cols, int idz, int rows_kernel,
                                                     int cols_kernel);

template __global__ void local_mean_kernel_3d<int>(int* image, float* output, float weight,
                                                   int rows, int cols, int depth, int rows_kernel,
                                                   int cols_kernel, int depth_kernel);
template __global__ void local_mean_kernel_3d<float>(float* image, float* output, float weight,
                                                     int rows, int cols, int depth, int rows_kernel,
                                                     int cols_kernel, int depth_kernel);

template <typename dtype>
void local_mean_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel) {

  dtype* dev_image;
  float* dev_output;

  cudaMalloc((void**)&dev_image, rows * cols * depth * sizeof(dtype));
  cudaMalloc((void**)&dev_output, rows * cols * depth * sizeof(float));

  cudaMemcpy(dev_image, image, rows * cols * depth * sizeof(dtype), cudaMemcpyHostToDevice);

  if (depth_kernel == 1) {

    dim3 blockSize(32, 32);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int idz = 0; idz < depth; ++idz) {
      local_mean_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, weight, rows, cols, idz,
                                                    rows_kernel, cols_kernel);

      cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  }

  else {

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x,
                  (depth + blockSize.z - 1) / blockSize.z);

    auto start = std::chrono::high_resolution_clock::now();

    local_mean_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, weight, rows, cols, depth,
                                                  rows_kernel, cols_kernel, depth_kernel);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  }

  cudaMemcpy(output, dev_output, rows * cols * depth * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_image);
  cudaFree(dev_output);
}

template void local_mean_threshold<float>(float* image, float* output, float weight, int rows,
                                          int cols, int depth, int rows_kernel, int cols_kernel,
                                          int depth_kernel);
template void local_mean_threshold<int>(int* image, float* output, float weight, int rows, int cols,
                                        int depth, int rows_kernel, int cols_kernel,
                                        int depth_kernel);
template void local_mean_threshold<unsigned int>(unsigned int* image, float* output, float weight,
                                                 int rows, int cols, int depth, int rows_kernel,
                                                 int cols_kernel, int depth_kernel);