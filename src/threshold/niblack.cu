#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/threshold/niblack.h"

/*

    based one: https://craftofcoding.wordpress.com/2021/09/30/thresholding-algorithms-niblack-local/

*/

template <typename dtype>
__global__ void niblack_kernel_2d(dtype* image, float* output, float weight, int rows, int cols,
                                  int idz, int rows_kernel, int cols_kernel) {

  //threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < rows && idy < cols) {
    //mean value
    float mean = 0;

    //standard deviation
    float standard_deviation = 0;

    //get the mean value
    get_mean_kernel_2d(image + idz * rows * cols, &mean, idx, idy, rows, cols, rows_kernel,
                       cols_kernel);

    //get the standard deviation
    get_std_kernel_2d(image + idz * rows * cols, mean, &standard_deviation, idx, idy, rows, cols,
                      rows_kernel, cols_kernel);

    //apply niblack threshold: T_{niblack} (i,j) = mean(i,j) - w * std(i,j)
    //threshold value
    float T_niblack = mean - weight * standard_deviation;

    if (image[idz * rows * cols + idx * cols + idy] > T_niblack) {
      output[idz * rows * cols + idx * cols + idy] = 255;

      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template <typename dtype>
__global__ void niblack_kernel_3d(dtype* image, float* output, float weight, int rows, int cols,
                                  int depth, int rows_kernel, int cols_kernel, int depth_kernel) {

  //threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < rows && idy < cols && idz < depth) {

    //mean value
    float mean = 0;

    //standard deviation
    float standard_deviation = 0;

    //get the mean value
    get_mean_kernel_3d(image, &mean, idx, idy, idz, rows, cols, depth, rows_kernel, cols_kernel,
                       depth_kernel);

    get_std_kernel_3d(image, mean, &standard_deviation, idx, idy, idz, rows, cols, depth,
                      rows_kernel, cols_kernel, depth_kernel);

    //apply niblack threshold: T_{niblack} (i,j,k) = mean(i,j,k) - w * std(i,j,k)
    //threshold value
    float T_niblack = mean - weight * standard_deviation;

    if (image[idz * rows * cols + idx * cols + idy] > T_niblack) {
      output[idz * rows * cols + idx * cols + idy] = 255;

      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template __global__ void niblack_kernel_2d<int>(int* image, float* output, float weight, int rows,
                                                int cols, int idz, int rows_kernel,
                                                int cols_kernel);
template __global__ void niblack_kernel_2d<float>(float* image, float* output, float weight,
                                                  int rows, int cols, int idz, int rows_kernel,
                                                  int cols_kernel);

template __global__ void niblack_kernel_3d<int>(int* image, float* output, float weight, int rows,
                                                int cols, int depth, int rows_kernel,
                                                int cols_kernel, int depth_kernel);
template __global__ void niblack_kernel_3d<float>(float* image, float* output, float weight,
                                                  int rows, int cols, int depth, int rows_kernel,
                                                  int cols_kernel, int depth_kernel);

template <typename dtype>
void niblack_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
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
      niblack_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, weight, rows, cols, idz,
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

    niblack_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, weight, rows, cols, depth,
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

template void niblack_threshold<float>(float* image, float* output, float weight, int rows,
                                       int cols, int depth, int rows_kernel, int cols_kernel,
                                       int depth_kernel);
template void niblack_threshold<int>(int* image, float* output, float weight, int rows, int cols,
                                     int depth, int rows_kernel, int cols_kernel, int depth_kernel);
template void niblack_threshold<unsigned int>(unsigned int* image, float* output, float weight,
                                              int rows, int cols, int depth, int rows_kernel,
                                              int cols_kernel, int depth_kernel);
