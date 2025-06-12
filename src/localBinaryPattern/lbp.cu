#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/localBinaryPattern/lbp.h"

// Reflect padding device function
__device__ int reflect_lbp(int idx, int limit) {
    if (idx < 0) return -idx;
    if (idx >= limit) return 2 * limit - idx - 1;
    return idx;
}

template <typename in_dtype>
__global__ void lbp(in_dtype* devImage, float* devOutput,
                    int xsize, int ysize, int zsize, int idz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= xsize || idy >= ysize)
        return;

    // Coordinates for this thread
    const int z_offset = idz * xsize * ysize;
    const int center_x = idx;
    const int center_y = idy;
    const int center_index = z_offset + center_x * ysize + center_y;
    const in_dtype center = devImage[center_index];

    // 8-neighbor LBP with reflection padding
    int dx[8] = {-1, -1, -1,  0, 1, 1, 1,  0};
    int dy[8] = {-1,  0,  1,  1, 1, 0, -1, -1};

    unsigned int code = 0;

    for (int k = 0; k < 8; ++k) {
        int neigh_x = reflect_lbp(center_x + dx[k], xsize);
        int neigh_y = reflect_lbp(center_y + dy[k], ysize);
        int neigh_index = z_offset + neigh_x * ysize + neigh_y;
        if (devImage[neigh_index] > center) {
            code |= 1 << (7 - k);  // now valid: integer bitwise op
        }
    }

    // Write output (same size as input)
    int out_index = idz * xsize * ysize + idx * ysize + idy;
    devOutput[out_index] = code;
}

// Explicit instantiation of the lbp kernel
template __global__ void lbp<int>(int* devImage, float* devOutput, int xsize, int ysize, int zsize, int idz);
template __global__ void lbp<unsigned int>(unsigned int* devImage, float* devOutput, int xsize, int ysize, int zsize, int idz);
template __global__ void lbp<float>(float* devImage, float* devOutput, int xsize, int ysize, int zsize, int idz);


template <typename in_dtype>
void localBinaryPattern(in_dtype* hostImage, float* hostOutput, int xsize, int ysize, int zsize)
{

  in_dtype* dev_image;
  float* dev_output;

  cudaMalloc((void**)&dev_image, xsize * ysize * zsize * sizeof(in_dtype));
  cudaMalloc((void**)&dev_output, xsize * ysize * zsize * sizeof(float));

  cudaMemcpy(dev_image, hostImage, xsize * ysize * zsize * sizeof(in_dtype), cudaMemcpyHostToDevice);

  dim3 blockSize(32, 32);
  dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

  for (int k = 0; k < zsize; ++k)
  {
         lbp<in_dtype><<<gridSize, blockSize>>>(dev_image, dev_output, xsize, ysize, zsize, k);

        cudaDeviceSynchronize();
  }

  cudaMemcpy(hostOutput, dev_output, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_image);
  cudaFree(dev_output);

}

// Explicit template instantiations
template void localBinaryPattern<float>(float*, float*, int, int, int);
template void localBinaryPattern<int>(int* , float* , int, int, int);
template void localBinaryPattern<unsigned int>(unsigned int*, float*, int, int, int);

