#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/log_filter.h"
#include "../../include/common/chunkedExecutor.h"

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
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    float temp;

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    convolution2d(image + idz * xsize * ysize, &temp, deviceKernel, idx, idy, xsize, ysize, 3, 3);

    output[index] = (float)sqrtf(temp * temp);
  }
}

template <typename dtype>
__global__ void log_filter_kernel_3d(dtype* image, float* output, float* deviceKernel, int xsize,
                                     int ysize, int zsize) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;

  //change xsize and ysize notation-->you made a mistake dummy.
  if (idx < xsize && idy < ysize && idz < zsize) {
    float temp;

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    convolution3d(image, &temp, deviceKernel, idx, idy, idz, xsize, ysize, zsize, 3, 3, 3);

    output[index] = (float)sqrtf(temp * temp);
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
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage, size * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, size * sizeof(float));

  cudaMemcpy(deviceImage, image, size * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    float* kernel;
    get_laplacian_kernel_2d(&kernel);

    float* deviceKernel;
    cudaMalloc((void**)&deviceKernel, 9 * sizeof(float));
    cudaMemcpy(deviceKernel, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

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
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y,
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

  cudaMemcpy(output, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);

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

//chunked executor version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void logFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize)
{
  const int paddedZsize = padding_bottom + zsize + padding_top;
  const unsigned int totalSize = xsize * ysize * paddedZsize;
  const int offset = padding_bottom * xsize * ysize;

  in_dtype* deviceImage;
  out_dtype* deviceOutput;
  cudaMalloc((void**)&deviceImage, totalSize * sizeof(in_dtype));
  cudaMalloc((void**)&deviceOutput, totalSize * sizeof(out_dtype));

  cudaMemcpy(deviceImage, hostImage, totalSize * sizeof(in_dtype), cudaMemcpyHostToDevice);

  kernel_dtype* deviceKernel;
  cudaMalloc((void**)&deviceKernel, kernel_xsize * kernel_ysize * kernel_zsize * sizeof(kernel_dtype));
  cudaMemcpy(deviceKernel, kernel, kernel_xsize * kernel_ysize * kernel_zsize * sizeof(kernel_dtype), cudaMemcpyHostToDevice);

  dim3 block(8, 8, 8);
  if (zsize == 1)
    block = dim3(32, 32, 1);

  dim3 grid((xsize + block.x - 1) / block.x,
            (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  if (verbose == 1) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  log_filter_kernel_3d<<<grid, block>>>(
      deviceImage + offset,  // start at first valid (unpadded) slice
      deviceOutput + offset,
      deviceKernel,
      xsize, ysize, zsize);

  cudaDeviceSynchronize();
  cudaMemcpy(hostOutput, deviceOutput + offset, xsize * ysize * zsize * sizeof(out_dtype), cudaMemcpyDeviceToHost);

  cudaFree(deviceKernel);
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

template void logFilter3DGPU<float, float, float>(float*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);
template void logFilter3DGPU<int, float, float>(int*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);
template void logFilter3DGPU<unsigned int, float, float>(unsigned int*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);


template<typename in_dtype, typename out_dtype>
void logFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize,
                      const int verbose, int ngpus, const float safetyMargin)
{
  if (ngpus == 0) {
    throw std::runtime_error("CPU implementation is not available for anisotropicDiffusion3D.");
  } else {
    int ncopies = 1;
    const int kernelOperations = 1;
    float* kernel;
    get_laplacian_kernel_3d(&kernel);  // kernel should be 3x3x3

    chunkedExecutorKernel(logFilter3DGPU<in_dtype, out_dtype, float>,
                          ncopies, safetyMargin, ngpus, kernelOperations,
                          hostImage, hostOutput, xsize, ysize, zsize, verbose,
                          kernel, 3, 3, 3);
  }
}

template void logFilterChunked<float, float>(float*, float*, const int, const int, const int, const int, int, const float);
template void logFilterChunked<int, float>(int*, float*, const int, const int, const int, const int, int, const float);
template void logFilterChunked<unsigned int, float>(unsigned int*, float*, const int, const int, const int, const int, int, const float);
