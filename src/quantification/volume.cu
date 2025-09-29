#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/quantification/volume.h"

__device__ void isVolume(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize,
                         int ysize, int zsize) {
  int imageIndex = idz * xsize * ysize + idy * xsize + idx;
  int counterIndex = image[imageIndex];

  atomicAdd(&counter[counterIndex], 1);
}

__global__ void volume_counter(int* image, unsigned int* counter, int idz, int xsize, int ysize,
                               int zsize) {
  // To compute the area, we just need to make the accumulated sum.
  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int idy = (threadIdx.y + blockIdx.y * blockDim.y);

  if (idx < xsize && idy < ysize) {
    isVolume(image, counter, idx, idy, idz, xsize, ysize, zsize);
  }
}

void volume(int* image, unsigned int* output, int xsize, int ysize, int zsize) {
  int* deviceImage;
  unsigned int* deviceOutput;

  cudaMalloc(&deviceImage, xsize * ysize * zsize * sizeof(int));
  cudaMalloc(&deviceOutput, xsize * ysize * zsize * sizeof(unsigned int));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(deviceOutput, 0,
             xsize * ysize * zsize * sizeof(unsigned int));  // Initialize output array to zero

  dim3 blockDim(32, 32);
  dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y);

  for (int idz = 0; idz < zsize; idz++) {
    volume_counter<<<gridDim, blockDim>>>(deviceImage, deviceOutput, idz, xsize, ysize, zsize);
  }

  cudaMemcpy(output, deviceOutput, xsize * ysize * zsize * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

/*
int main()
{
    const int xsize = 8;
    const int ysize = 8;
    const int zsize = 8;

    unsigned int output[xsize * ysize * zsize] = {0};

    int hostImage[xsize * ysize * zsize] = 
    {
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
    };

    volume(hostImage, output, xsize, ysize, zsize);

    for (int i = 0; i < ysize; i++) {
        for (int j = 0; j < xsize; j++) {
            std::cout << output[i*xsize +j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/