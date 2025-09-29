#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/mean_filter.h"
#include "../../include/common/chunkedExecutor.h"

template <typename dtype>
__global__ void mean_filter_kernel_2d(dtype* image, float* output, int xsize, int ysize, int idz,
                                      int nx, int ny) {

  //threads
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    //mean value
    float mean = 0;

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    //get the neighbors
    get_mean_kernel_2d(image + idz * xsize * ysize, &mean, idx, idy, xsize, ysize, nx, ny);

    //assign the mean value
    output[index] = mean;
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

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    //get the neighbors
    get_mean_kernel_3d(image, &mean, idx, idy, idz, xsize, ysize, zsize, nx, ny, nz);

    //assign the mean value
    output[index] = mean;
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
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage, size * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, size * sizeof(float));

  cudaMemcpy(deviceImage, image, size * sizeof(dtype), cudaMemcpyHostToDevice);

  if (nz == 1) {

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      mean_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, xsize, ysize, k, nx,
                                                     ny);

      cudaDeviceSynchronize();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

  }

  else {

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y,
                  (zsize + blockSize.z - 1) / blockSize.z);

    //auto start = std::chrono::high_resolution_clock::now();

    mean_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, xsize, ysize, zsize,
                                                   nx, ny, nz);

    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  }

  cudaMemcpy(output, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);

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


//chunked executor variant
template <typename in_dtype, typename out_dtype>
void meanFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                     int nx, int ny, int nz)
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

  mean_filter_kernel_3d<<<grid, block>>>(deviceImage, deviceOutput, xsize, ysize, zsize,
                                                   nx, ny, nz);

  cudaDeviceSynchronize();

  cudaMemcpy(hostOutput, deviceOutput, size * sizeof(out_dtype), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);

}

// Explicit instantiation for float
template void meanFilter3DGPU<float, float>(float* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int flag_verbose,
                                            int nx, int ny, int nz);

template void meanFilter3DGPU<int, float>(int* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int flag_verbose,
                                            int nx, int ny, int nz);

template void meanFilter3DGPU<unsigned int, float>(unsigned int* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int flag_verbose,
                                            int nx, int ny, int nz);

template<typename in_dtype, typename out_dtype>
void meanFilterChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int type3d, int flag_verbose,
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
    mean_filtering(hostImage, hostOutput,xsize,ysize,zsize,nx,ny,1);
    std::cout<<"2d variant\n";

  }

  else
  {

    int ncopies = 2;
    chunkedExecutor(meanFilter3DGPU<in_dtype,out_dtype>, ncopies, gpuMemory, ngpus,
                    hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, nx, ny, nz);

  }

}

template void meanFilterChunked<float, float>(float* hostImage, float* hostOutput,
                                              int xsize, int ysize, int zsize, int type3d, int flag_verbose,
                                              float gpuMemory, int ngpus, int nx, int ny, int nz);

template void meanFilterChunked<int, float>(int* hostImage, float* hostOutput,
                                            int xsize, int ysize, int zsize, int type3d,int flag_verbose,
                                            float gpuMemory, int ngpus, int nx, int ny, int nz);

template void meanFilterChunked<unsigned int, float>(unsigned int* hostImage, float* hostOutput,
                                                     int xsize, int ysize, int zsize, int type3d,int flag_verbose,
                                                     float gpuMemory, int ngpus, int nx, int ny, int nz);
