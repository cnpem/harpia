#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/sobel_filter.h"
#include "../../include/common/chunkedExecutor.h"

void get_sobel_horizontal_kernel_2d(float** kernel) {
  /*

        Horizontal hostKernel has the form:

                   1  0 -1
                   2  0 -2
                   1  0 -1

    */

  *kernel = (float*)malloc(sizeof(float) * 9);

  if (!*kernel) {
    return;
  }

  (*kernel)[0] = 1;
  (*kernel)[1] = 0;
  (*kernel)[2] = -1;

  (*kernel)[3] = 2;
  (*kernel)[4] = 0;
  (*kernel)[5] = -2;

  (*kernel)[6] = 1;
  (*kernel)[7] = 0;
  (*kernel)[8] = -1;
}

void get_sobel_vertical_kernel_2d(float** kernel) {
  /*

        Vertical hostKernel has the form:

                   1  2  1
                   0  0  0
                  -1 -2 -1

    */

  *kernel = (float*)malloc(sizeof(float) * 9);

  if (!*kernel) {
    return;
  }

  (*kernel)[0] = 1;
  (*kernel)[1] = 2;
  (*kernel)[2] = 1;

  (*kernel)[3] = 0;
  (*kernel)[4] = 0;
  (*kernel)[5] = 0;

  (*kernel)[6] = -1;
  (*kernel)[7] = -2;
  (*kernel)[8] = -1;
}

void get_sobel_horizontal_kernel_3d(float** kernel) {
  //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/


  //kernel allocation.
  *kernel = (float*)malloc(27 * sizeof(float));

  if (!*kernel) {
    return;
  }

  //first plane
  (*kernel)[0] = -1;
  (*kernel)[1] = 0;
  (*kernel)[2] = 1;

  (*kernel)[3] = -2;
  (*kernel)[4] = 0;
  (*kernel)[5] = 2;

  (*kernel)[6] = -1;
  (*kernel)[7] = 0;
  (*kernel)[8] = 1;

  //second plane
  (*kernel)[9] = -2;
  (*kernel)[10] = 0;
  (*kernel)[11] = -2;

  (*kernel)[12] = -4;
  (*kernel)[13] = 0;
  (*kernel)[14] = 4;

  (*kernel)[15] = -2;
  (*kernel)[16] = 0;
  (*kernel)[17] = 2;

  //third plane
  (*kernel)[18] = -1;
  (*kernel)[19] = 0;
  (*kernel)[20] = 1;

  (*kernel)[21] = -2;
  (*kernel)[22] = 0;
  (*kernel)[23] = 2;

  (*kernel)[24] = -1;
  (*kernel)[25] = 0;
  (*kernel)[26] = 1;
}

void get_sobel_vertical_kernel_3d(float** kernel) {

  //kernel allocation.
  *kernel = (float*)malloc(27 * sizeof(float));

  if (!*kernel) {
    return;
  }

  //first plane
  (*kernel)[0] = -1;
  (*kernel)[1] = -2;
  (*kernel)[2] = -1;

  (*kernel)[3] = 0;
  (*kernel)[4] = 0;
  (*kernel)[5] = 0;

  (*kernel)[6] = 1;
  (*kernel)[7] = 2;
  (*kernel)[8] = 1;

  //second plane
  (*kernel)[9] = -2;
  (*kernel)[10] = -4;
  (*kernel)[11] = -2;

  (*kernel)[12] = 0;
  (*kernel)[13] = 0;
  (*kernel)[14] = 0;

  (*kernel)[15] = 2;
  (*kernel)[16] = 4;
  (*kernel)[17] = 2;

  //third plane
  (*kernel)[18] = -1;
  (*kernel)[19] = -2;
  (*kernel)[20] = -1;

  (*kernel)[21] = 0;
  (*kernel)[22] = 0;
  (*kernel)[23] = 0;

  (*kernel)[24] = 1;
  (*kernel)[25] = 2;
  (*kernel)[26] = 1;
}

void get_sobel_depth_kernel_3d(float** kernel) {

  //kernel allocation.
  *kernel = (float*)malloc(27 * sizeof(float));

  if (!*kernel) {
    return;
  }

  //first plane
  (*kernel)[0] = -1;
  (*kernel)[1] = -2;
  (*kernel)[2] = -1;

  (*kernel)[3] = -2;
  (*kernel)[4] = -4;
  (*kernel)[5] = -2;

  (*kernel)[6] = -1;
  (*kernel)[7] = -2;
  (*kernel)[8] = -1;

  //second plane
  (*kernel)[9] = 0;
  (*kernel)[10] = 0;
  (*kernel)[11] = 0;

  (*kernel)[12] = 0;
  (*kernel)[13] = 0;
  (*kernel)[14] = 0;

  (*kernel)[15] = 0;
  (*kernel)[16] = 0;
  (*kernel)[17] = 0;

  //third plane
  (*kernel)[18] = 1;
  (*kernel)[19] = 2;
  (*kernel)[20] = 1;

  (*kernel)[21] = 2;
  (*kernel)[22] = 4;
  (*kernel)[23] = 2;

  (*kernel)[24] = 1;
  (*kernel)[25] = 2;
  (*kernel)[26] = 1;
}

template <typename dtype>
__global__ void sobel_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
                                       float* deviceKernelVertical, int idz, int xsize, int ysize,
                                       int zsize) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    float tempVertical;
    float tempHorizontal;

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    convolution2d(image + idz * xsize * ysize, &tempHorizontal, deviceKernelHorizontal, idx, idy,
                  xsize, ysize, 3, 3);
    convolution2d(image + idz * xsize * ysize, &tempVertical, deviceKernelVertical, idx, idy, xsize,
                  ysize, 3, 3);

    output[index] = tempHorizontal * tempHorizontal + tempVertical * tempVertical;
    output[index] = (float)sqrtf(output[index]);
  }
}

template <typename dtype>
__global__ void sobel_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                       float* deviceKernelVertical, float* deviceKernelDepth,
                                       int xsize, int ysize, int zsize) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    float tempVertical;
    float tempHorizontal;
    float tempDepth;

    unsigned int index = idz * xsize * ysize + idx * ysize + idy;

    convolution3d(image, &tempHorizontal, deviceKernelHorizontal, idx, idy, idz, xsize, ysize,
                  zsize, 3, 3, 3);
    convolution3d(image, &tempVertical, deviceKernelVertical, idx, idy, idz, xsize, ysize, zsize, 3,
                  3, 3);
    convolution3d(image, &tempDepth, deviceKernelDepth, idx, idy, idz, xsize, ysize, zsize, 3, 3,
                  3);

    output[index] = tempHorizontal * tempHorizontal + tempVertical * tempVertical + tempDepth * tempDepth;
    output[index] = (float)sqrtf(output[index]);
  }
}

template __global__ void sobel_filter_kernel_2d<int>(int* image, float* output,
                                                     float* deviceKernelHorizontal,
                                                     float* deviceKernelVertical, int idz,
                                                     int xsize, int ysize, int zsize);
template __global__ void sobel_filter_kernel_2d<float>(float* image, float* output,
                                                       float* deviceKernelHorizontal,
                                                       float* deviceKernelVertical, int idz,
                                                       int xsize, int ysize, int zsize);

template __global__ void sobel_filter_kernel_3d<int>(int* image, float* output,
                                                     float* deviceKernelHorizontal,
                                                     float* deviceKernelVertical,
                                                     float* deviceKernelDepth, int xsize, int ysize,
                                                     int zsize);
template __global__ void sobel_filter_kernel_3d<float>(float* image, float* output,
                                                       float* deviceKernelHorizontal,
                                                       float* deviceKernelVertical,
                                                       float* deviceKernelDepth, int xsize,
                                                       int ysize, int zsize);

template <typename dtype>
void sobel_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type) {

  dtype* deviceImage;
  float* deviceOutput;
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage,size * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput,size * sizeof(float));

  cudaMemcpy(deviceImage, image,size * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    float* kernelHorizontal;
    get_sobel_horizontal_kernel_2d(&kernelHorizontal);

    float* kernelVertical;
    get_sobel_vertical_kernel_2d(&kernelVertical);

    float* deviceKernelHorizontal;
    cudaMalloc((void**)&deviceKernelHorizontal, 9 * sizeof(float));
    cudaMemcpy(deviceKernelHorizontal, kernelHorizontal, 9 * sizeof(float), cudaMemcpyHostToDevice);

    float* deviceKernelVertical;
    cudaMalloc((void**)&deviceKernelVertical, 9 * sizeof(float));
    cudaMemcpy(deviceKernelVertical, kernelVertical, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      sobel_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput,
                                                      deviceKernelHorizontal, deviceKernelVertical,
                                                      k, xsize, ysize, zsize);
    }
    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernelHorizontal);
    cudaFree(deviceKernelVertical);
  }

  else {
    float* kernelHorizontal;
    get_sobel_horizontal_kernel_3d(&kernelHorizontal);

    float* kernelVertical;
    get_sobel_vertical_kernel_3d(&kernelVertical);

    float* kernelDepth;
    get_sobel_depth_kernel_3d(&kernelDepth);  // Corrected function name

    float* deviceKernelHorizontal;
    cudaMalloc((void**)&deviceKernelHorizontal, 27 * sizeof(float));
    cudaMemcpy(deviceKernelHorizontal, kernelHorizontal, 27 * sizeof(float),
               cudaMemcpyHostToDevice);

    float* deviceKernelVertical;
    cudaMalloc((void**)&deviceKernelVertical, 27 * sizeof(float));
    cudaMemcpy(deviceKernelVertical, kernelVertical, 27 * sizeof(float), cudaMemcpyHostToDevice);

    float* deviceKernelDepth;
    cudaMalloc((void**)&deviceKernelDepth, 27 * sizeof(float));
    cudaMemcpy(deviceKernelDepth, kernelDepth, 27 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y,
                  (zsize + blockSize.z - 1) / blockSize.z);

    //auto start = std::chrono::high_resolution_clock::now();

    sobel_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput,
                                                    deviceKernelHorizontal, deviceKernelVertical,
                                                    deviceKernelDepth, xsize, ysize, zsize);

    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernelHorizontal);
    cudaFree(deviceKernelVertical);
    cudaFree(deviceKernelDepth);
  }

  cudaMemcpy(output, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

// Explicit instantiation
template void sobel_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize,
                                     bool type);
template void sobel_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize,
                                   bool type);
template void sobel_filtering<unsigned int>(unsigned int* image, float* output, int xsize,
                                            int ysize, int zsize, bool type);


// chunked version
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void sobelFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
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

  float* kernelHorizontal;
  get_sobel_horizontal_kernel_3d(&kernelHorizontal);

  float* kernelVertical;
  get_sobel_vertical_kernel_3d(&kernelVertical);

  float* kernelDepth;
  get_sobel_depth_kernel_3d(&kernelDepth);

  float* deviceKernelHorizontal;
  cudaMalloc((void**)&deviceKernelHorizontal, 27 * sizeof(float));
  cudaMemcpy(deviceKernelHorizontal, kernelHorizontal, 27 * sizeof(float), cudaMemcpyHostToDevice);

  float* deviceKernelVertical;
  cudaMalloc((void**)&deviceKernelVertical, 27 * sizeof(float));
  cudaMemcpy(deviceKernelVertical, kernelVertical, 27 * sizeof(float), cudaMemcpyHostToDevice);

  float* deviceKernelDepth;
  cudaMalloc((void**)&deviceKernelDepth, 27 * sizeof(float));
  cudaMemcpy(deviceKernelDepth, kernelDepth, 27 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(8, 8, 8);
  if (zsize == 1) block = dim3(32, 32, 1);

  dim3 grid((xsize + block.x - 1) / block.x,
            (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  if (verbose == 1) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  sobel_filter_kernel_3d<<<grid, block>>>(
      deviceImage + offset,
      deviceOutput + offset,
      deviceKernelHorizontal,
      deviceKernelVertical,
      deviceKernelDepth,
      xsize, ysize, zsize);

  cudaDeviceSynchronize();
  cudaMemcpy(hostOutput, deviceOutput + offset, xsize * ysize * zsize * sizeof(out_dtype), cudaMemcpyDeviceToHost);

  cudaFree(deviceKernelHorizontal);
  cudaFree(deviceKernelVertical);
  cudaFree(deviceKernelDepth);
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

// Explicit template instantiations
template void sobelFilter3DGPU<float, float, float>(float*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);
template void sobelFilter3DGPU<int, float, float>(int*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);
template void sobelFilter3DGPU<unsigned int, float, float>(unsigned int*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);


template<typename in_dtype, typename out_dtype>
void sobelFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                          const int xsize, const int ysize, const int zsize, const int type3d,
                          const int verbose, int ngpus, const float safetyMargin)
{
  if (ngpus == 0) {
    throw std::runtime_error("CPU implementation is not available for sobelFilter3D.");
  }
  
  else if (zsize==1 || type3d == 0)
  {
    //calls 2d variant
    sobel_filtering(hostImage, hostOutput,xsize,ysize,zsize,0);
    std::cout<<"2d variant\n";

  }
  
  else {
    int ncopies = 1;
    const int kernelOperations = 1;
    float* dummyKernel = nullptr;  // not used, but placeholder for compatibility

    chunkedExecutorKernel(sobelFilter3DGPU<in_dtype, out_dtype, float>,
                          ncopies, safetyMargin, ngpus, kernelOperations,
                          hostImage, hostOutput, xsize, ysize, zsize, verbose,
                          dummyKernel, 3, 3, 3);
  }
}

// Explicit instantiations
template void sobelFilterChunked<float, float>(float*, float*, const int, const int, const int, const int,const int, int, const float);
template void sobelFilterChunked<int, float>(int*, float*, const int, const int, const int, const int,const int, int, const float);
template void sobelFilterChunked<unsigned int, float>(unsigned int*, float*, const int, const int, const int, const int,const int, int, const float);

/*

int main()
{
    int xsize = 1024/2;
    int ysize = 1024/2;
    int zsize = 1024/2;

    static int* deviceImage;
    deviceImage = (int*)malloc(zsize*xsize*ysize*sizeof(int));

    static int* deviceOutput;
    deviceOutput = (int*)malloc(zsize*xsize*ysize*sizeof(int));

    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                if (i!=j)
                {
                    deviceImage[k * xsize * ysize + i * ysize + j] = 1;
                }

                if (i==j)
                {
                    deviceImage[k * xsize * ysize + i * ysize + j] = -1;
                }
                
        
                deviceOutput[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }


    sobel_filtering(deviceImage,deviceOutput,xsize,ysize,zsize, false);
    
    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                std::cout<<deviceOutput[k*xsize*ysize + i*ysize +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    

    

    return 0;
}

*/