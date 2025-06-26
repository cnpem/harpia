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

//for chunked version
template <typename dtype>
__global__ void local_gaussian_kernel_3d_chunked(dtype* image, float* output, double* dev_kernel,
                                         float weight, int rows, int cols, int depth,
                                         int rows_kernel, int cols_kernel, int depth_kernel,
                                         int padding_bottom, int padding_top) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < rows && idy < cols && idz < depth) {
    double temp;

    convolution3d_chunked(image, &temp, dev_kernel, idx, idy, idz, rows, cols, depth,padding_bottom,padding_top, rows_kernel,
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

    //auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < slices; ++k) {
      local_gaussian_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, weight,
                                                        k, rows, cols, slices, rows_kernel,
                                                        cols_kernel);
    }
    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

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

    //auto start = std::chrono::high_resolution_clock::now();

    local_gaussian_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, weight,
                                                      rows, cols, slices, rows_kernel, cols_kernel,
                                                      depth_kernel);

    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

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
                                      const int ysize, const int zsize, const int flag_verbose,
                                      int padding_bottom, int padding_top,
                                      kernel_dtype* kernel,
                                      int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                      float weight)
{
  const int paddedZsize = padding_bottom + zsize + padding_top;
  const size_t totalSize = static_cast<size_t>(xsize) * ysize * paddedZsize;
  const size_t offset = static_cast<size_t>(padding_bottom) * xsize * ysize;

  size_t inputBytes = totalSize * sizeof(in_dtype);
  size_t outputBytes = static_cast<size_t>(xsize) * ysize * zsize * sizeof(out_dtype);
  size_t kernelBytes = kernel_xsize * kernel_ysize * kernel_zsize * sizeof(kernel_dtype);

  in_dtype *i_deviceImage, *deviceImage, *i_hostImage;
  out_dtype* deviceOutput;
  kernel_dtype* deviceKernel;

  // 1. Allocate device memory
  CHECK(cudaMalloc((void**)&i_deviceImage, inputBytes));
  CHECK(cudaMalloc((void**)&deviceOutput, outputBytes));
  CHECK(cudaMalloc((void**)&deviceKernel, kernelBytes));

  // 2. Adjust host image pointer to include padding
  i_hostImage = hostImage - offset;

  // 3. Copy data to device
  CHECK(cudaMemcpy(i_deviceImage, i_hostImage, inputBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernelBytes, cudaMemcpyHostToDevice));

  // 4. Set the image pointer to the non-padded region
  deviceImage = i_deviceImage + offset;

  // 5. Configure CUDA grid and block dimensions
  dim3 block(8, 8, 8);
  if (zsize == 1)
    block = dim3(32, 32, 1);

  dim3 grid((xsize + block.x - 1) / block.x,
            (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  if (flag_verbose) {
    std::cout << "grid: (" << grid.x << "," << grid.y << "," << grid.z << ")\n";
    std::cout << "block: (" << block.x << "," << block.y << "," << block.z << ")\n";
  }

  // 6. Launch kernel
  local_gaussian_kernel_3d<<<grid, block>>>(
      deviceImage, deviceOutput, deviceKernel, weight,
      xsize, ysize, zsize,
      kernel_xsize, kernel_ysize, kernel_zsize);

  CHECK(cudaDeviceSynchronize());

  // 7. Copy result back to host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, outputBytes, cudaMemcpyDeviceToHost));

  // 8. Free memory
  cudaFree(i_deviceImage);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
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
    int ncopies = 2;
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
