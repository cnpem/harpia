#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/prewitt_filter.h"

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
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < xsize && idy < ysize) {
    float tempVertical;
    float tempHorizontal;

    convolution2d(image + idz * xsize * ysize, &tempHorizontal, deviceKernelHorizontal, idx, idy,
                  xsize, ysize, 3, 3);
    convolution2d(image + idz * xsize * ysize, &tempVertical, deviceKernelVertical, idx, idy, xsize,
                  ysize, 3, 3);

    output[idz * xsize * ysize + idx * ysize + idy] =
        tempHorizontal * tempHorizontal + tempVertical * tempVertical;
    output[idz * xsize * ysize + idx * ysize + idy] =
        (float)sqrtf(output[idz * xsize * ysize + idx * ysize + idy]);
  }
}

template <typename dtype>
__global__ void prewitt_filter_kernel_3d(dtype* image, float* output, float* deviceKernelHorizontal,
                                         float* deviceKernelVertical, float* deviceKernelDepth,
                                         int xsize, int ysize, int zsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    float tempVertical;
    float tempHorizontal;
    float tempDepth;

    convolution3d(image, &tempHorizontal, deviceKernelHorizontal, idx, idy, idz, xsize, ysize,
                  zsize, 3, 3, 3);
    convolution3d(image, &tempVertical, deviceKernelVertical, idx, idy, idz, xsize, ysize, zsize, 3,
                  3, 3);
    convolution3d(image, &tempDepth, deviceKernelDepth, idx, idy, idz, xsize, ysize, zsize, 3, 3,
                  3);

    output[idz * xsize * ysize + idx * ysize + idy] =
        tempHorizontal * tempHorizontal + tempVertical * tempVertical + tempDepth * tempDepth;
    output[idz * xsize * ysize + idx * ysize + idy] =
        (float)sqrtf(output[idz * xsize * ysize + idx * ysize + idy]);
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
  cudaMalloc((void**)&deviceImage, xsize * ysize * zsize * sizeof(dtype));
  cudaMalloc((void**)&deviceOutput, xsize * ysize * zsize * sizeof(float));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

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

    dim3 blockSize(16, 16);
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x);

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
    dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x,
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

  cudaMemcpy(output, deviceOutput, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

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