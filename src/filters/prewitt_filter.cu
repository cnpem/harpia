#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/prewitt_filter.h"
#include "../../include/common/chunkedExecutor.h"

void get_prewitt_horizontal_kernel_2d(float** kernel) {
  //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

  /*
        +--------------+
        |   1   0  -1  |
        |   1   0  -1  |
        |   1   0  -1  |
        +--------------+
    
    */

  *kernel = (float*)malloc(sizeof(float) * 9);

  if (!*kernel) {
    return;
  }

  (*kernel)[0] = 1;
  (*kernel)[1] = 0;
  (*kernel)[2] = -1;

  (*kernel)[3] = 1;
  (*kernel)[4] = 0;
  (*kernel)[5] = -1;

  (*kernel)[6] = 1;
  (*kernel)[7] = 0;
  (*kernel)[8] = -1;
}

void get_prewitt_vertical_kernel_2d(float** kernel) {
  //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

  /*

        +--------------+
        |   1   1   1  |
        |   0   0   0  |
        |  -1  -1  -1  |
        +--------------+


    */

  *kernel = (float*)malloc(sizeof(float) * 9);

  if (!*kernel) {
    return;
  }

  (*kernel)[0] = 1;
  (*kernel)[1] = 1;
  (*kernel)[2] = 1;

  (*kernel)[3] = 0;
  (*kernel)[4] = 0;
  (*kernel)[5] = 0;

  (*kernel)[6] = -1;
  (*kernel)[7] = -1;
  (*kernel)[8] = -1;
}

void get_prewitt_horizontal_kernel_3d(float** kernel) {
  //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

  /*
            
                 +---------------+
                /    -1  0  1   /|
               /     -1  0  1  / |
              /      -1  0  1 /  |
             +---------------+   |
            /    -1  0  1   /|  /
           /     -1  0  1  / | /
          /      -1  0  1 /  |/ 
         +---------------+   +
         |    -1  0  1   |  /
         |    -1  0  1   | /
         |    -1  0  1   |/
         +---------------+

    
    */

  //kernel allocation.
  *kernel = (float*)malloc(27 * sizeof(float));

  if (!*kernel) {
    return;
  }

  //first plane
  (*kernel)[0] = -1;
  (*kernel)[1] = 0;
  (*kernel)[2] = 1;

  (*kernel)[3] = -1;
  (*kernel)[4] = 0;
  (*kernel)[5] = 1;

  (*kernel)[6] = -1;
  (*kernel)[7] = 0;
  (*kernel)[8] = 1;

  //second plane
  (*kernel)[9] = -1;
  (*kernel)[10] = 0;
  (*kernel)[11] = 1;

  (*kernel)[12] = -1;
  (*kernel)[13] = 0;
  (*kernel)[14] = 1;

  (*kernel)[15] = -1;
  (*kernel)[16] = 0;
  (*kernel)[17] = 1;

  //third plane
  (*kernel)[18] = -1;
  (*kernel)[19] = 0;
  (*kernel)[20] = 1;

  (*kernel)[21] = -1;
  (*kernel)[22] = 0;
  (*kernel)[23] = 1;

  (*kernel)[24] = -1;
  (*kernel)[25] = 0;
  (*kernel)[26] = 1;
}

void get_prewitt_vertical_kernel_3d(float** kernel) {
  //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

  /*
            
                 +---------------+
                /  -1  -1  -1   /|
               /    0   0   0  / |
              /     1   1   1 /  |
             +---------------+   |
            /  -1  -1  -1   /|  /
           /    0   0   0  / | /
          /     1   1   1 /  |/ 
         +---------------+   +
         |   -1  -1  -1  |  /
         |    0   0   0  | /
         |    1   1   1  |/
         +---------------+

    
    */

  //kernel allocation.
  *kernel = (float*)malloc(27 * sizeof(float));

  if (!*kernel) {
    return;
  }

  //first plane
  (*kernel)[0] = -1;
  (*kernel)[1] = -1;
  (*kernel)[2] = -1;

  (*kernel)[3] = 0;
  (*kernel)[4] = 0;
  (*kernel)[5] = 0;

  (*kernel)[6] = 1;
  (*kernel)[7] = 1;
  (*kernel)[8] = 1;

  //second plane
  (*kernel)[9] = -1;
  (*kernel)[10] = -1;
  (*kernel)[11] = -1;

  (*kernel)[12] = 0;
  (*kernel)[13] = 0;
  (*kernel)[14] = 0;

  (*kernel)[15] = 1;
  (*kernel)[16] = 1;
  (*kernel)[17] = 1;

  //third plane
  (*kernel)[18] = -1;
  (*kernel)[19] = -1;
  (*kernel)[20] = -1;

  (*kernel)[21] = 0;
  (*kernel)[22] = 0;
  (*kernel)[23] = 0;

  (*kernel)[24] = 1;
  (*kernel)[25] = 1;
  (*kernel)[26] = 1;
}

void get_prewitt_depth_kernel_3d(float** kernel) {
  //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

  /*
            
                 +---------------+
                /   1   1   1   /|
               /    1   1   1  / |
              /     1   1   1 /  |
             +---------------+   |
            /   0   0   0   /|  /
           /    0   0   0  / | /
          /     0   0   0 /  |/ 
         +---------------+   +
         |   -1  -1  -1  |  /
         |   -1  -1  -1  | /
         |   -1  -1  -1  |/
         +---------------+

    
    */

  // Kernel allocation
  *kernel = (float*)malloc(27 * sizeof(float));

  if (!*kernel) {
    return;
  }

  // First plane
  (*kernel)[0] = 1;
  (*kernel)[1] = 1;
  (*kernel)[2] = 1;

  (*kernel)[3] = 1;
  (*kernel)[4] = 1;
  (*kernel)[5] = 1;

  (*kernel)[6] = 1;
  (*kernel)[7] = 1;
  (*kernel)[8] = 1;

  // Second plane
  (*kernel)[9] = 0;
  (*kernel)[10] = 0;
  (*kernel)[11] = 0;

  (*kernel)[12] = 0;
  (*kernel)[13] = 0;
  (*kernel)[14] = 0;

  (*kernel)[15] = 0;
  (*kernel)[16] = 0;
  (*kernel)[17] = 0;

  // Third plane
  (*kernel)[18] = -1;
  (*kernel)[19] = -1;
  (*kernel)[20] = -1;

  (*kernel)[21] = -1;
  (*kernel)[22] = -1;
  (*kernel)[23] = -1;

  (*kernel)[24] = -1;
  (*kernel)[25] = -1;
  (*kernel)[26] = -1;
}

template <typename dtype>
__global__ void prewitt_filter_kernel_2d(dtype* image, float* output, float* deviceKernelHorizontal,
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
__global__ void prewitt_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
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

template __global__ void prewitt_filter_kernel_2d<int>(int* image, float* output,
                                                       float* deviceKernelHorizontal,
                                                       float* deviceKernelVertical, int idz,
                                                       int xsize, int ysize, int zsize);
template __global__ void prewitt_filter_kernel_2d<float>(float* image, float* output,
                                                         float* deviceKernelHorizontal,
                                                         float* deviceKernelVertical, int idz,
                                                         int xsize, int ysize, int zsize);

template __global__ void prewitt_filter_kernel_3d<int>(int* image, float* output,
                                                       float* deviceKernelHorizontal,
                                                       float* deviceKernelVertical,
                                                       float* deviceKernelDepth, int xsize,
                                                       int ysize, int zsize);
template __global__ void prewitt_filter_kernel_3d<float>(float* image, float* output,
                                                         float* deviceKernelHorizontal,
                                                         float* deviceKernelVertical,
                                                         float* deviceKernelDepth, int xsize,
                                                         int ysize, int zsize);

template <typename dtype>
void prewitt_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type) {

  dtype* deviceImage;
  float* deviceOutput;
  unsigned int size = xsize * ysize * zsize;

  cudaMalloc((void**)&deviceImage, size * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, size * sizeof(float));

  cudaMemcpy(deviceImage, image, size * sizeof(dtype), cudaMemcpyHostToDevice);

  if (type == false) {
    float* kernelHorizontal;
    get_prewitt_horizontal_kernel_2d(&kernelHorizontal);

    float* kernelVertical;
    get_prewitt_vertical_kernel_2d(&kernelVertical);

    float* deviceKernelHorizontal;
    cudaMalloc((void**)&deviceKernelHorizontal, 9 * sizeof(float));
    cudaMemcpy(deviceKernelHorizontal, kernelHorizontal, 9 * sizeof(float), cudaMemcpyHostToDevice);

    float* deviceKernelVertical;
    cudaMalloc((void**)&deviceKernelVertical, 9 * sizeof(float));
    cudaMemcpy(deviceKernelVertical, kernelVertical, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < zsize; ++k) {
      prewitt_filter_kernel_2d<<<gridSize, blockSize>>>(
          deviceImage, deviceOutput, deviceKernelHorizontal, deviceKernelVertical, k, xsize, ysize,
          zsize);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernelHorizontal);
    cudaFree(deviceKernelVertical);
  }

  else {
    float* kernelHorizontal;
    get_prewitt_horizontal_kernel_3d(&kernelHorizontal);

    float* kernelVertical;
    get_prewitt_vertical_kernel_3d(&kernelVertical);

    float* kernelDepth;
    get_prewitt_depth_kernel_3d(&kernelDepth);  // Corrected function name

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

    auto start = std::chrono::high_resolution_clock::now();

    prewitt_filter_kernel_3d<<<gridSize, blockSize>>>(deviceImage, deviceOutput,
                                                      deviceKernelHorizontal, deviceKernelVertical,
                                                      deviceKernelDepth, xsize, ysize, zsize);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(deviceKernelHorizontal);
    cudaFree(deviceKernelVertical);
    cudaFree(deviceKernelDepth);
  }

  cudaMemcpy(output, deviceOutput, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

// Explicit instantiation
template void prewitt_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize,
                                       bool type);
template void prewitt_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize,
                                     bool type);
template void prewitt_filtering<unsigned int>(unsigned int* image, float* output, int xsize,
                                              int ysize, int zsize, bool type);


template <typename in_dtype, typename out_dtype, typename kernel_dtype>
void prewittFilter3DGPU(in_dtype* hostImage, out_dtype* hostOutput, const int xsize,
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
  get_prewitt_horizontal_kernel_3d(&kernelHorizontal);

  float* kernelVertical;
  get_prewitt_vertical_kernel_3d(&kernelVertical);

  float* kernelDepth;
  get_prewitt_depth_kernel_3d(&kernelDepth);

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

  prewitt_filter_kernel_3d<<<grid, block>>>(
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
template void prewittFilter3DGPU<float, float, float>(float*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);
template void prewittFilter3DGPU<int, float, float>(int*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);
template void prewittFilter3DGPU<unsigned int, float, float>(unsigned int*, float*, const int, const int, const int, const int, int, int, float*, int, int, int);


template<typename in_dtype, typename out_dtype>
void prewittFilterChunked(in_dtype* hostImage, out_dtype* hostOutput,
                          const int xsize, const int ysize, const int zsize,
                          const int verbose, int ngpus, const float safetyMargin)
{
  if (ngpus == 0) {
    throw std::runtime_error("CPU implementation is not available for prewittFilter3D.");
  } else {
    int ncopies = 1;
    const int kernelOperations = 1;
    float* dummyKernel = nullptr;  // not used, but placeholder for compatibility

    chunkedExecutorKernel(prewittFilter3DGPU<in_dtype, out_dtype, float>,
                          ncopies, safetyMargin, ngpus, kernelOperations,
                          hostImage, hostOutput, xsize, ysize, zsize, verbose,
                          dummyKernel, 3, 3, 3);
  }
}

// Explicit instantiations
template void prewittFilterChunked<float, float>(float*, float*, const int, const int, const int, const int, int, const float);
template void prewittFilterChunked<int, float>(int*, float*, const int, const int, const int, const int, int, const float);
template void prewittFilterChunked<unsigned int, float>(unsigned int*, float*, const int, const int, const int, const int, int, const float);


/*

int main()
{
    int xsize = 10;
    int ysize = 10;
    int zsize = 1;

    static int* image;
    image = (int*)malloc(zsize*xsize*ysize*sizeof(int));

    static int* output;
    output = (int*)malloc(zsize*xsize*ysize*sizeof(int));

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
                    image[k * xsize * ysize + i * ysize + j] = -1;
                }
                
        
                output[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }


    prewitt_filtering(image,output,xsize,ysize,zsize, false);
    
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