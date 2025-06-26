#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/gaussian_filter.h"
#include "../../include/common/chunkedExecutor.h"

template <typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output, double* deviceKernel, int idz,
                                          int xsize, int ysize, int zsize, int nx, int ny) {

  //threads indices
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  // general matrix convolution for each pixel of the image.
  if (idx < xsize && idy < ysize) {
    //temp variable
    double temp;

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    //convolution.
    convolution2d(image + idz * xsize * ysize, &temp, deviceKernel, idx, idy, xsize, ysize, nx, ny);

    output[index] = (double)temp;
  }
}

template <typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output, double* deviceKernel,
                                          int xsize, int ysize, int zsize, int nx, int ny, int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    double temp;
    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    convolution3d(image, &temp, deviceKernel, idx, idy, idz, xsize, ysize, zsize, nx, ny, nz);

    output[index] = (double)temp;
  }
}

//used in the chunked version
template <typename dtype>
__global__ void gaussian_filter_kernel_3d_chunked(dtype* image, float* output, double* deviceKernel,
                                     int xsize, int ysize, int zsize, int nx, int ny, int nz,
                                     int padding_bottom, int padding_top) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    double temp;

    const size_t index = static_cast<size_t>(idz) * xsize * ysize +
                         static_cast<size_t>(idx) * ysize +
                         static_cast<size_t>(idy);

    // Apply convolution on padded memory
    convolution3d_chunked(image, &temp, deviceKernel,
                               idx, idy, idz,
                               xsize, ysize, zsize,
                               padding_bottom, padding_top,
                               nx, ny, nz);

    output[index] = sqrtf(temp * temp);
  }
}

template __global__ void gaussian_filter_kernel_2d<int>(int* image, float* output,
                                                        double* deviceKernel, int idz, int xsize,
                                                        int ysize, int zsize, int nx, int ny);
template __global__ void gaussian_filter_kernel_2d<float>(float* image, float* output,
                                                          double* deviceKernel, int idz, int xsize,
                                                          int ysize, int zsize, int nx, int ny);

template __global__ void gaussian_filter_kernel_3d<int>(int* image, float* output,
                                                        double* deviceKernel, int xsize, int ysize,
                                                        int zsize, int nx, int ny, int nz);
template __global__ void gaussian_filter_kernel_3d<float>(float* image, float* output,
                                                          double* deviceKernel, int xsize, int ysize,
                                                          int zsize, int nx, int ny, int nz);

template __global__ void gaussian_filter_kernel_3d_chunked<int>(int* image, float* output,
                                                        double* deviceKernel, int xsize, int ysize,
                                                        int zsize, int nx, int ny, int nz,
                                                        int padding_bottom, int padding_top);

template __global__ void gaussian_filter_kernel_3d_chunked<float>(float* image, float* output,
                                                          double* deviceKernel, int xsize, int ysize,
                                                          int zsize, int nx, int ny, int nz,
                                                          int padding_bottom, int padding_top);


template <typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma,
                        bool type) {

  dtype* deviceImage;
  float* deviceOutput;
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage, size * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput,size * sizeof(float));

  cudaMemcpy(deviceImage, image, size * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    //kernel size
    int nx = (int)ceil(4 * sigma + 0.5);
    int ny = nx;

    double* kernel;
    get_gaussian_kernel_2d(&kernel, nx, ny, sigma);

    double* deviceKernel;
    cudaMalloc((void**)&deviceKernel, nx * ny * sizeof(double));
    cudaMemcpy(deviceKernel, kernel, nx * ny * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel, k,
                                                         xsize, ysize, zsize, nx, ny);

      cudaDeviceSynchronize();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernel);
    free(kernel);
  }

  else {
    //kernel size
    int nx = (int)ceil(4 * sigma + 0.5);
    int ny = nx;
    int nz = nx;

    double* kernel;
    get_gaussian_kernel_3d(&kernel, nx, ny, nz, sigma);

    double* deviceKernel;
    cudaMalloc((void**)&deviceKernel, nx * ny * nz * sizeof(double));
    cudaMemcpy(deviceKernel, kernel, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y,
                  (zsize + blockSize.z - 1) / blockSize.z);

    //auto start = std::chrono::high_resolution_clock::now();

    gaussian_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel,
                                                       xsize, ysize, zsize, nx, ny, nz);

    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernel);
    free(kernel);
  }

  cudaMemcpy(output, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

// Explicit instantiation
template void gaussian_filtering<float>(float* image, float* output, int xsize, int ysize,
                                        int zsize, float sigma, bool type);
template void gaussian_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize,
                                      float sigma, bool type);
template void gaussian_filtering<unsigned int>(unsigned int* image, float* output, int xsize,
                                               int ysize, int zsize, float sigma, bool type);


//chunked executor version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void gaussianFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
                         const int ysize, const int zsize, const int flag_verbose,
                         int padding_bottom, int padding_top,
                         kernel_dtype* kernel,
                         int kernel_xsize, int kernel_ysize, int kernel_zsize)
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

  // Allocate memory
  CHECK(cudaMalloc((void**)&i_deviceImage, inputBytes));
  CHECK(cudaMalloc((void**)&deviceOutput, outputBytes));
  CHECK(cudaMalloc((void**)&deviceKernel, kernelBytes));

  // Offset host pointer for padding
  i_hostImage = hostImage - offset;

  // Copy padded image and kernel to device
  CHECK(cudaMemcpy(i_deviceImage, i_hostImage, inputBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernelBytes, cudaMemcpyHostToDevice));

  // Adjust device pointer to the start of the unpadded region
  deviceImage = i_deviceImage + offset;

  // Define grid and block
  dim3 block(8, 8, 8);
  if (zsize == 1) block = dim3(32, 32, 1);

  dim3 grid((xsize + block.x - 1) / block.x,
            (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  if (flag_verbose) {
    std::cout << "grid: (" << grid.x << "," << grid.y << "," << grid.z << ")\n";
    std::cout << "block: (" << block.x << "," << block.y << "," << block.z << ")\n";
  }

  // Launch kernel
  gaussian_filter_kernel_3d<<<grid, block>>>(
      deviceImage,
      deviceOutput,
      deviceKernel,
      xsize, ysize, zsize,
      kernel_xsize, kernel_ysize, kernel_zsize);

  CHECK(cudaDeviceSynchronize());

  // Copy output to host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, outputBytes, cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(i_deviceImage);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}

template void gaussianFilter3DGPU<float, float, double>(float*, float*, const int, const int, const int, const int, int, int, double*, int, int, int);
template void gaussianFilter3DGPU<int, float, double>(int*, float*, const int, const int, const int, const int, int, int, double*, int, int, int);
template void gaussianFilter3DGPU<unsigned int, float, double>(unsigned int*, float*, const int, const int, const int, const int, int, int, double*, int, int, int);


template<typename in_dtype, typename out_dtype>
void gaussianFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                      const int xsize, const int ysize, const int zsize, float sigma, const int type3d,
                      const int verbose, int ngpus,const float safetyMargin )
{
  if (ngpus == 0) {
    throw std::runtime_error("CPU implementation is not available for anisotropicDiffusion3D.");
  }
  
  else if (zsize==1 || type3d == 0)
  {
    //calls 2d variant
    gaussian_filtering(hostImage, hostOutput,xsize,ysize,zsize,sigma,0);
    std::cout<<"2d variant\n";

  }

  
  else {
    int ncopies = 2;
    const int kernelOperations = 1;
    double* kernel;
    int gaussian_size = (int)ceil(4 * sigma + 0.5); 
    get_gaussian_kernel_3d(&kernel,gaussian_size,gaussian_size,gaussian_size,sigma); 

    chunkedExecutorKernel(gaussianFilter3DGPU<in_dtype, out_dtype, double>,
                          ncopies, safetyMargin, ngpus, kernelOperations,
                          hostImage, hostOutput, xsize, ysize, zsize, verbose,
                          kernel, gaussian_size, gaussian_size, gaussian_size);

  free(kernel);
  }
}

template void gaussianFilterChunked<float, float>(float*, float*, const int, const int, const int, float,const int, const int, int, const float);
template void gaussianFilterChunked<int, float>(int*, float*, const int, const int, const int, float,const int, const int, int, const float);
template void gaussianFilterChunked<unsigned int, float>(unsigned int*, float*, const int, const int, const int, float,const int, const int, int, const float);


/*
int main()
{
    int xsize = 512;
    int ysize = 512;
    int zsize = 512;

    static float* image;
    image = (float*)malloc(zsize*xsize*ysize*sizeof(int));

    static float* output;
    output = (float*)malloc(zsize*xsize*ysize*sizeof(int));

    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                if (i!=j)
                {
                    image[k * xsize * ysize + i * ysize + j] = 1;
                }

                if (i==j)
                {
                    image[k * xsize * ysize + i * ysize + j] = i+j;
                }
                
        
                output[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }

    float sigma = 6.;
    gaussian_filtering(image,output,xsize,ysize,zsize,sigma,true);
    
    
    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                std::cout<<output[k*xsize*ysize + i*ysize +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    

    

    return 0;
}
*/