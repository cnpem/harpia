#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/threshold/adaptative_gaussian.h"
#include "../../include/common/chunkedExecutor.h"

template <typename dtype>
__global__ void local_gaussian_kernel_2d(dtype* image, float* output, double* dev_kernel,
                                         float weight, int idz, int rows, int cols, int slices,
                                         int rows_kernel, int cols_kernel) {

  //threads indices
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  // general matrix convolution for each pixel of the image.
  if (idx < rows && idy < cols) {
    //temp variable
    double temp;

    //convolution.
    convolution2d(image + idz * rows * cols, &temp, dev_kernel, idx, idy, rows, cols, rows_kernel,
                  cols_kernel);

    double T_local_gaussian = temp - weight;

    if (image[idz * rows * cols + idx * cols + idy] > T_local_gaussian) {
      output[idz * rows * cols + idx * cols + idy] = 255;
      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template <typename dtype>
__global__ void local_gaussian_kernel_3d(dtype* image, float* output, double* dev_kernel,
                                         float weight, int rows, int cols, int depth,
                                         int rows_kernel, int cols_kernel, int depth_kernel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < rows && idy < cols && idz < depth) {
    double temp;

    convolution3d(image, &temp, dev_kernel, idx, idy, idz, rows, cols, depth, rows_kernel,
                  cols_kernel, depth_kernel);

    double T_local_gaussian = temp - weight;

    if (image[idz * rows * cols + idx * cols + idy] > T_local_gaussian) {
      output[idz * rows * cols + idx * cols + idy] = 255;
      return;
    }

    output[idz * rows * cols + idx * cols + idy] = 0;
  }
}

template __global__ void local_gaussian_kernel_2d<int>(int* image, float* output, double* dev_kernel,
                                                       float weight, int idz, int rows, int cols,
                                                       int slices, int rows_kernel,
                                                       int cols_kernel);
template __global__ void local_gaussian_kernel_2d<float>(float* image, float* output,
                                                         double* dev_kernel, float weight, int idz,
                                                         int rows, int cols, int slices,
                                                         int rows_kernel, int cols_kernel);

template __global__ void local_gaussian_kernel_3d<int>(int* image, float* output, double* dev_kernel,
                                                       float weight, int rows, int cols, int depth,
                                                       int rows_kernel, int cols_kernel,
                                                       int depth_kernel);
template __global__ void local_gaussian_kernel_3d<float>(float* image, float* output,
                                                         double* dev_kernel, float weight, int rows,
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
    int rows_kernel = (int)ceil(6 * sigma + 1);
    int cols_kernel = rows_kernel;

    double* kernel;
    get_gaussian_kernel_2d(&kernel, rows_kernel, cols_kernel, sigma);

    double* dev_kernel;
    cudaMalloc((void**)&dev_kernel, rows_kernel * cols_kernel * sizeof(double));
    cudaMemcpy(dev_kernel, kernel, rows_kernel * cols_kernel * sizeof(double),
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
    int rows_kernel = (int)ceil(6 * sigma + 1);
    int cols_kernel = rows_kernel;
    int depth_kernel = rows_kernel;

    double* kernel;
    get_gaussian_kernel_3d(&kernel, rows_kernel, cols_kernel, depth_kernel, sigma);

    double* dev_kernel;
    cudaMalloc((void**)&dev_kernel, rows_kernel * cols_kernel * depth_kernel * sizeof(double));
    cudaMemcpy(dev_kernel, kernel, rows_kernel * cols_kernel * depth_kernel * sizeof(double),
               cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
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


//chunked executor version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void adaptativeGaussianThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                    const int ysize, const int zsize, const int verbose,
                    int padding_bottom, int padding_top,
                    kernel_dtype* kernel,
                    int kernel_xsize, int kernel_ysize, int kernel_zsize, float weight)
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

  local_gaussian_kernel_3d<<<grid, block>>>(deviceImage+ offset,
                                             deviceOutput+ offset,
                                             deviceKernel,weight,
                                             xsize, ysize, zsize, 
                                             kernel_xsize, kernel_ysize, kernel_zsize);

  cudaDeviceSynchronize();
  cudaMemcpy(hostOutput, deviceOutput + offset, xsize * ysize * zsize * sizeof(out_dtype), cudaMemcpyDeviceToHost);

  cudaFree(deviceKernel);
  cudaFree(deviceImage);
  cudaFree(deviceOutput);

}

template void adaptativeGaussianThreshold3DGPU<float, float, double>(float*, float*, const int, const int, const int, const int, int, int, double*, int, int, int, float);
template void adaptativeGaussianThreshold3DGPU<int, float, double>(int*, float*, const int, const int, const int, const int, int, int, double*, int, int, int, float);
template void adaptativeGaussianThreshold3DGPU<unsigned int, float, double>(unsigned int*, float*, const int, const int, const int, const int, int, int, double*, int, int, int, float);

template<typename in_dtype, typename out_dtype>
void adaptativeGaussianThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize, float sigma, float weight, const int type3d,
                      const int verbose, int ngpus,const float safetyMargin )
{
  if (ngpus == 0) {
    throw std::runtime_error("CPU implementation is not available for anisotropicDiffusion3D.");
  }
  
  else if (zsize==1 || type3d == 0)
  {
    //calls 2d variant
    local_gaussian_threshold(hostImage, hostOutput,xsize,ysize,zsize,sigma,weight,0);
    std::cout<<"2d variant\n";

  }

  
  else {
    int ncopies = 1;
    const int kernelOperations = 1;
    double* kernel;
    int gaussian_size = (int)ceil(6 * sigma + 1); 
    get_gaussian_kernel_3d(&kernel,gaussian_size,gaussian_size,gaussian_size,sigma); 

    chunkedExecutorKernel(adaptativeGaussianThreshold3DGPU<in_dtype, out_dtype, double>,
                          ncopies, safetyMargin, ngpus, kernelOperations,
                          hostImage, hostOutput, xsize, ysize, zsize, verbose,
                          kernel, gaussian_size, gaussian_size, gaussian_size,weight);
  }
}

template void adaptativeGaussianThresholdChunked<float, float>(float*, float*, const int, const int, const int, float,float,const int, const int, int, const float);
template void adaptativeGaussianThresholdChunked<int, float>(int*, float*, const int, const int, const int, float,float,const int, const int, int, const float);
template void adaptativeGaussianThresholdChunked<unsigned int, float>(unsigned int*, float*, const int, const int, const int,float, float,const int, const int, int, const float);
