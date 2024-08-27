#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/threshold/adaptative_gaussian.h"

template <typename dtype>
__global__ void local_gaussian_kernel_2d(dtype* image, float* output, float* dev_kernel,
                                         float weight, int idz, int rows, int cols, int slices,
                                         int rows_kernel, int cols_kernel) {

  //threads indices
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  // general matrix convolution for each pixel of the image.
  if (idx < rows && idy < cols) {
    //temp variable
    float temp;

    //convolution.
    convolution2d(image + idz * rows * cols, &temp, dev_kernel, idx, idy, rows, cols, rows_kernel,
                  cols_kernel);

    float T_local_gaussian = temp - weight;

    if (image[idz * rows * cols + idx * cols + idy] > T_local_gaussian) {
      output[idz * rows * cols + idx * cols + idy] = 255;
      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template <typename dtype>
__global__ void local_gaussian_kernel_3d(dtype* image, float* output, float* dev_kernel,
                                         float weight, int rows, int cols, int depth,
                                         int rows_kernel, int cols_kernel, int depth_kernel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < rows && idy < cols && idz < depth) {
    float temp;

    convolution3d(image, &temp, dev_kernel, idx, idy, idz, rows, cols, depth, rows_kernel,
                  cols_kernel, depth_kernel);

    float T_local_gaussian = temp - weight;

    if (image[idz * rows * cols + idx * cols + idy] > T_local_gaussian) {
      output[idz * rows * cols + idx * cols + idy] = 255;
      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template __global__ void local_gaussian_kernel_2d<int>(int* image, float* output, float* dev_kernel,
                                                       float weight, int idz, int rows, int cols,
                                                       int slices, int rows_kernel,
                                                       int cols_kernel);
template __global__ void local_gaussian_kernel_2d<float>(float* image, float* output,
                                                         float* dev_kernel, float weight, int idz,
                                                         int rows, int cols, int slices,
                                                         int rows_kernel, int cols_kernel);

template __global__ void local_gaussian_kernel_3d<int>(int* image, float* output, float* dev_kernel,
                                                       float weight, int rows, int cols, int depth,
                                                       int rows_kernel, int cols_kernel,
                                                       int depth_kernel);
template __global__ void local_gaussian_kernel_3d<float>(float* image, float* output,
                                                         float* dev_kernel, float weight, int rows,
                                                         int cols, int depth, int rows_kernel,
                                                         int cols_kernel, int depth_kernel);

template <typename dtype>
void local_gaussian_threshold(dtype* image, float* output, int rows, int cols, int slices,
                              float sigma, float weight, bool type) {

  dtype* dev_image;
  float* dev_output;
  cudaMalloc((void**)&dev_image, rows * cols * slices * sizeof(dtype));
  cudaMalloc((void**)&dev_output, rows * cols * slices * sizeof(float));

  cudaMemcpy(dev_image, image, rows * cols * slices * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    //kernel size
    int rows_kernel = (int)ceil(2 * sigma + 1);
    int cols_kernel = rows_kernel;

    float* kernel;
    get_gaussian_kernel_2d(&kernel, rows_kernel, cols_kernel, sigma);

    float* dev_kernel;
    cudaMalloc((void**)&dev_kernel, rows_kernel * cols_kernel * sizeof(float));
    cudaMemcpy(dev_kernel, kernel, rows_kernel * cols_kernel * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < slices; ++k) {
      local_gaussian_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, weight,
                                                        k, rows, cols, slices, rows_kernel,
                                                        cols_kernel);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(dev_kernel);
  }

  else {
    //kernel size
    int rows_kernel = (int)ceil(2 * sigma + 1);
    int cols_kernel = rows_kernel;
    int depth_kernel = rows_kernel;

    float* kernel;
    get_gaussian_kernel_3d(&kernel, rows_kernel, cols_kernel, depth_kernel, sigma);

    float* dev_kernel;
    cudaMalloc((void**)&dev_kernel, rows_kernel * cols_kernel * depth_kernel * sizeof(float));
    cudaMemcpy(dev_kernel, kernel, rows_kernel * cols_kernel * depth_kernel * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 4);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x,
                  (slices + blockSize.z - 1) / blockSize.z);

    auto start = std::chrono::high_resolution_clock::now();

    local_gaussian_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, weight,
                                                      rows, cols, slices, rows_kernel, cols_kernel,
                                                      depth_kernel);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(dev_kernel);
  }

  cudaMemcpy(output, dev_output, rows * cols * slices * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_image);
  cudaFree(dev_output);
}

template void local_gaussian_threshold<float>(float* image, float* output, int rows, int cols,
                                              int slices, float sigma, float weight, bool type);
template void local_gaussian_threshold<int>(int* image, float* output, int rows, int cols,
                                            int slices, float sigma, float weight, bool type);
template void local_gaussian_threshold<unsigned int>(unsigned int* image, float* output, int rows,
                                                     int cols, int slices, float sigma,
                                                     float weight, bool type);