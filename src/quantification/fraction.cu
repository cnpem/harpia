#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/quantification/fraction.h"

__global__ void fraction_counter(int* image, int* counter, int acumulator, int xsize, int ysize,
                                 int zsize) {

  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int idy = (threadIdx.y + blockIdx.y * blockDim.y);
  int idz = (threadIdx.z + blockIdx.z * blockDim.z);

  if (idx < xsize && idy < ysize && idz < zsize) {
    int imageIndex = idz * xsize * ysize + idy * xsize + idx;
    int counterIndex = image[imageIndex];

    atomicAdd(&counter[counterIndex], 1);

    atomicAdd(&acumulator, 1);
  }
}

__global__ void labels_fraction(int* image, int* counter, int acumulator, int xsize, int ysize,
                                int zsize) {

  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int idy = (threadIdx.y + blockIdx.y * blockDim.y);
  int idz = (threadIdx.z + blockIdx.z * blockDim.z);

  if (idx < xsize && idy < ysize && idz < zsize) {
    int index = idz * xsize * ysize + idy * xsize + idx;

    counter[index] = 100 * counter[index] / acumulator;
  }
}

void fraction(int* image, int* output, int xsize, int ysize, int zsize) {

  int* deviceImage;
  int* deviceOutput;
  int* deviceAcumulator;

  cudaMalloc(&deviceImage, xsize * ysize * zsize * sizeof(int));
  cudaMalloc(&deviceOutput, xsize * ysize * zsize * sizeof(int));
  cudaMalloc(&deviceAcumulator, 1 * sizeof(int));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);

  dim3 blockDim(8, 8, 8);  // Example block dimensions, can be adjusted
  dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y,
               (zsize + blockDim.z - 1) / blockDim.z);

  fraction_counter<<<gridDim, blockDim>>>(deviceImage, deviceOutput, *deviceAcumulator, xsize,
                                          ysize, zsize);

  labels_fraction<<<gridDim, blockDim>>>(deviceImage, deviceOutput, *deviceAcumulator, xsize, ysize,
                                         zsize);

  cudaMemcpy(output, deviceOutput, xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(image, deviceImage, xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}
