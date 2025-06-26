#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/threshold/niblack.h"
#include "../../include/common/chunkedExecutor.h"

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

    //auto start = std::chrono::high_resolution_clock::now();

    for (int idz = 0; idz < depth; ++idz) {
      niblack_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, weight, rows, cols, idz,
                                                 rows_kernel, cols_kernel);

      cudaDeviceSynchronize();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  }

  else {

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x,
                  (depth + blockSize.z - 1) / blockSize.z);

    //auto start = std::chrono::high_resolution_clock::now();

    niblack_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, weight, rows, cols, depth,
                                               rows_kernel, cols_kernel, depth_kernel);

    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
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


//chunked version
template <typename in_dtype, typename out_dtype>
void niblackThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                     int nx, int ny, int nz, float weight)
{
  in_dtype* deviceImage;
  out_dtype* deviceOutput;
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage, size * sizeof(in_dtype));
  cudaMalloc((void**)&deviceOutput, size * sizeof(out_dtype));

  cudaMemcpy(deviceImage, hostImage, size * sizeof(in_dtype), cudaMemcpyHostToDevice);

  dim3 block(8, 8, 8);

  if (zsize == 1)
  {
    block = dim3(32, 32, 1);
  }

  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  if (flag_verbose==1) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }
  niblack_kernel_3d<<<grid, block>>>(deviceImage, deviceOutput, weight, xsize, ysize, zsize,
                                                   nx, ny, nz);

  cudaDeviceSynchronize();

  cudaMemcpy(hostOutput, deviceOutput, size * sizeof(out_dtype), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);

}

// Explicit instantiation for float
template void niblackThreshold3DGPU<float, float>(float* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int flag_verbose,
                                            int nx, int ny, int nz, float weight);

template void niblackThreshold3DGPU<int, float>(int* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int flag_verbose,
                                            int nx, int ny, int nz, float weight);

template void niblackThreshold3DGPU<unsigned int, float>(unsigned int* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int flag_verbose,
                                            int nx, int ny, int nz, float weight);


template<typename in_dtype, typename out_dtype>
void niblackThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,float weight, int type3d, int flag_verbose,
                       float gpuMemory, int ngpus, int nx, int ny, int nz)
{
  if (ngpus == 0)
  {
      throw std::runtime_error("CPU implementation is not available for anisotropicDiffusion3D. "
        "Please ensure a GPU is available to execute this function.");

  }

  else if (zsize==1 || type3d == 0 || nz == 1)
  {
    //calls 2d variant
    niblack_threshold(hostImage, hostOutput,weight,xsize,ysize,zsize,nx,ny,1);
    std::cout<<"2d variant\n";

  }

  else
  {

    int ncopies = 2;
    chunkedExecutor(niblackThreshold3DGPU<in_dtype,out_dtype>, ncopies, gpuMemory, ngpus,
                    hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, nx, ny, nz,weight);

  }

}

template void niblackThresholdChunked<float, float>(float* hostImage, float* hostOutput,
                                              int xsize, int ysize, int zsize, float weight, int type3d, int flag_verbose,
                                              float gpuMemory, int ngpus, int nx, int ny, int nz);

template void niblackThresholdChunked<int, float>(int* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, float weight,int type3d,int flag_verbose,
                                            float gpuMemory, int ngpus, int nx, int ny, int nz);

template void niblackThresholdChunked<unsigned int, float>(unsigned int* hostImage, float* hostOutput,
                                                     int xsize, int ysize, int zsize, float weight,int type3d,int flag_verbose,
                                                     float gpuMemory, int ngpus, int nx, int ny, int nz);
