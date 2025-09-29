#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/median_filter.h"

template <typename dtype>
__device__ void bubble_sort(dtype* array, int size) {

  int j;
  int flag = 1;

  dtype temp;

  if (!array || size < 2) {
    return;
  }

  while (flag != 0) {
    flag = 0;

    for (j = 0; j < size - 1; j++) {
      if (array[j] > array[j + 1]) {
        temp = array[j];

        array[j] = array[j + 1];

        array[j + 1] = temp;

        flag = 1;
      }
    }
  }
}

template <typename dtype>
__device__ void get_median_kernel_2d(dtype* image, dtype* kernel, int i, int j, int xsize,
                                     int ysize, int nx, int ny) {

  int inputY;
  int inputX;

  for (int m = 0; m < nx; m++) {

    for (int n = 0; n < ny; n++) {
      //this is needed to compute everything with respect to the center of the kernel.
      inputX = i - nx / 2 + m;
      inputY = j - ny / 2 + n;

      // Check if inputX and inputY are within bounds
      if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize) {
        kernel[(i * ysize + j) * (nx * ny) + (m * ny + n)] = image[inputX * ysize + inputY];
      }

      //make a padding function to substitute this line of code.
      else {

        // Reflect padding
        if (inputX < 0)
          inputX = -inputX;
        else if (inputX >= xsize)
          inputX = 2 * xsize - inputX - 1;

        if (inputY < 0)
          inputY = -inputY;
        else if (inputY >= ysize)
          inputY = 2 * ysize - inputY - 1;

        kernel[(i * ysize + j) * (nx * ny) + (m * ny + n)] = image[inputX * ysize + inputY];
      }
    }
  }
}

template <typename dtype>
__device__ void get_median_kernel_3d(dtype* image, dtype* kernel, int i, int j, int k, int xsize,
                                     int ysize, int zsize, int nx, int ny, int nz) {

  int inputY;
  int inputX;
  int inputZ;

  for (int l = 0; l < nz; l++) {

    for (int m = 0; m < nx; m++) {

      for (int n = 0; n < ny; n++) {
        //this is needed to compute everything with respect to the center of the kernel.
        inputX = i - nx / 2 + m;
        inputY = j - ny / 2 + n;
        inputZ = k - nz / 2 + l;

        if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize && inputZ >= 0 &&
            inputZ < zsize) {

          unsigned int index = (inputZ * xsize * ysize) + (inputX * ysize) + inputY;
          kernel[(k * xsize * ysize + i * ysize + j) * (nx * ny * nz) + (l * nx * ny) + (m * ny) +n] = image[index];
        }

        //make a padding function to substitute this line of code.
        else {
          // Reflect padding
          if (inputX < 0) {
            inputX = -inputX;
          }

          else if (inputX >= xsize) {
            inputX = 2 * xsize - inputX - 1;
          }

          if (inputY < 0) {
            inputY = -inputY;
          }

          else if (inputY >= ysize) {
            inputY = 2 * ysize - inputY - 1;
          }

          if (inputZ < 0) {
            inputZ = -inputZ;
          }

          else if (inputZ >= zsize) {
            inputZ = 2 * zsize - inputZ - 1;
          }

          unsigned int index = (inputZ * xsize * ysize) + (inputX * ysize) + inputY;
          kernel[(k * xsize * ysize + i * ysize + j) * (nx * ny * nz) + (l * nx * ny) + (m * ny) +n] = image[index];
        }
      }
    }
  }
}

template <typename dtype>
__global__ void median_filter_kernel_2d(dtype* image, dtype* output, dtype* kernel, int xsize,
                                        int ysize, int idz, int nx, int ny) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    get_median_kernel_2d(image + idz * xsize * ysize, kernel, idx, idy, xsize, ysize, nx, ny);
    bubble_sort(kernel + (idx * ysize + idy) * (nx * ny), nx * ny);

    int medianIndex = (nx * ny) / 2;

    if ((nx * ny) % 2 == 0) {
      dtype medianValue = (kernel[(idx * ysize + idy) * (nx * ny) + medianIndex] +
                           kernel[(idx * ysize + idy) * (nx * ny) + medianIndex - 1]) /
                          2;
      output[idz * xsize * ysize + idx * ysize + idy] = medianValue;
    }

    else {
      output[idz * xsize * ysize + idx * ysize + idy] =
          kernel[(idx * ysize + idy) * (nx * ny) + medianIndex];
    }
  }
}

template <typename dtype>
__global__ void median_filter_kernel_3d(dtype* image, dtype* output, dtype* kernel, int xsize,
                                        int ysize, int zsize, int idz, int nx, int ny, int nz) {

  //threads
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {

    unsigned int index = (idz * xsize * ysize + idx * ysize + idy) * (nx * ny * nz);
    unsigned int out_index = idz * xsize * ysize + idx * ysize + idy;

    get_median_kernel_3d(image, kernel, idx, idy, idz, xsize, ysize, zsize, nx, ny, nz);

    bubble_sort(kernel + index, nx * ny * nz);

    int medianIndex = (nx * ny * nz) / 2;

    if ((nx * ny * nz) % 2 == 0) 
    {
      dtype medianValue = (kernel[index + medianIndex] + kernel[index + medianIndex - 1]) / 2;

      output[out_index] = medianValue;
    } 
    
    else 
    
    {
      output[out_index] = kernel[index + medianIndex];
    }

  }
}

template __global__ void median_filter_kernel_2d<int>(int* image, int* output, int* kernel,
                                                      int xsize, int ysize, int idz, int nx,
                                                      int ny);
template __global__ void median_filter_kernel_2d<float>(float* image, float* output, float* kernel,
                                                        int xsize, int ysize, int idz, int nx,
                                                        int ny);

template __global__ void median_filter_kernel_3d<int>(int* deviceImage, int* deviceOutput, int* deviceKernel,
                                                      int xsize, int ysize, int zsize, int idz,
                                                      int nx, int ny, int nz);

template __global__ void median_filter_kernel_3d<float>(float* deviceImage, float* deviceOutput, float* deviceKernel,
                                                        int xsize, int ysize, int zsize, int idz,
                                                        int nx, int ny, int nz);

template <typename dtype>
void median_filtering(dtype* image, dtype* output, int xsize, int ysize, int zsize, int nx, int ny,
                      int nz) {

  dtype* deviceImage;
  dtype* deviceOutput;
  dtype* deviceKernel;
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage, size * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, size * sizeof(dtype));
  cudaMalloc((void**)&deviceKernel, xsize * ysize * nx * ny * nz * sizeof(dtype));

  cudaMemcpy(deviceImage, image, size * sizeof(dtype), cudaMemcpyHostToDevice);

  if (nz == 1) {

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      median_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel,
                                                       xsize, ysize, k, nx, ny);

      cudaDeviceSynchronize();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  }

  else {

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      median_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel,
                                                       xsize, ysize, zsize, k, nx, ny, nz);

      cudaDeviceSynchronize();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  }

  cudaMemcpy(output, deviceOutput, size * sizeof(dtype), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}

// Explicit instantiation for dtype
template void median_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize,
                                      int nx, int ny, int nz);
template void median_filtering<int>(int* image, int* output, int xsize, int ysize, int zsize,
                                    int nx, int ny, int nz);
template void median_filtering<unsigned int>(unsigned int* image, unsigned int* output, int xsize,
                                             int ysize, int zsize, int nx, int ny, int nz);