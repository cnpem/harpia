#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include "../../include/filters/anisotropic_diffusion.h"

template <typename dtype>
__global__ void anisotropicDiffusion2DKernel(dtype* hostImage, dtype* outputImage, float deltaT,
                                             float kappa, int diffusionOption, int xsize,
                                             int ysize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idy < xsize && idx < ysize) {

    // Compute indices for the neighboring cells with boundary checks
    int idyNorth = min(idy + 1, xsize - 1);
    int idySouth = max(idy - 1, 0);
    int idxEast = min(idx + 1, ysize - 1);
    int idxWest = max(idx - 1, 0);

    dtype center = hostImage[idy * ysize + idx];
    dtype nabla[8];
    double_t diffusionCoefficients[8];

    nabla[0] = hostImage[idyNorth * ysize + idx] - center;      // North
    nabla[1] = hostImage[idySouth * ysize + idx] - center;      // South
    nabla[2] = hostImage[idy * ysize + idxWest] - center;       // West
    nabla[3] = hostImage[idy * ysize + idxEast] - center;       // East
    nabla[4] = hostImage[idyNorth * ysize + idxWest] - center;  // Northwest
    nabla[5] = hostImage[idyNorth * ysize + idxEast] - center;  // Northeast
    nabla[6] = hostImage[idySouth * ysize + idxWest] - center;  // Southwest
    nabla[7] = hostImage[idySouth * ysize + idxEast] - center;  // Southeast

    double_t diffusionSum = 0;

    for (int i = 0; i < 8; i++) {
      double scaledDiff = pow(nabla[i] / kappa, 2);
      if (diffusionOption == 1) {
        diffusionCoefficients[i] = nabla[i] * exp(-scaledDiff);
      } else if (diffusionOption == 2) {
        diffusionCoefficients[i] = nabla[i] / (1 + scaledDiff);
      } else {
        diffusionCoefficients[i] = nabla[i] * (1 - tanh(scaledDiff));
      }

      diffusionSum += diffusionCoefficients[i];
    }

    outputImage[idy * ysize + idx] = hostImage[idy * ysize + idx] + deltaT * diffusionSum;
  }
}

template <typename dtype>
void anisotropicDiffusion2DGPU(dtype* hostImage, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize) {
  dtype *deviceImage, *deviceTmp;
  size_t numBytes = xsize * ysize * sizeof(dtype);

  // Allocate memory for the input image on the device
  cudaMalloc((void**)&deviceImage, numBytes);
  cudaMalloc((void**)&deviceTmp, numBytes);

  cudaMemcpy(deviceImage, hostImage, numBytes, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((ysize + blockSize.x - 1) / blockSize.x, (xsize + blockSize.y - 1) / blockSize.y);

  for (int iter = 0; iter < totalIterations; iter++) {

    anisotropicDiffusion2DKernel<dtype><<<gridSize, blockSize>>>(
        deviceImage, deviceTmp, deltaT, kappa, diffusionOption, xsize, ysize);

    cudaDeviceSynchronize();  // Synchronous barrier at each time step iteration

    std::swap(deviceImage, deviceTmp);
  }

  cudaMemcpy(hostImage, deviceImage, numBytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceTmp);
}

template void anisotropicDiffusion2DGPU<float>(float*, int, float, float, int, int, int);
template void anisotropicDiffusion2DGPU<double>(double*, int, float, float, int, int, int);

// on device, change name
template <typename dtype>
__global__ void anisotropicDiffusion3DKernel(dtype* hostImage, dtype* outputImage, float deltaT,
                                             float kappa, int diffusionOption, int xsize, int ysize,
                                             int zsize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idy < xsize && idx < ysize && idz < zsize) {

    int idx_center = idz * xsize * ysize + idy * ysize + idx;

    dtype center = hostImage[idx_center];
    double_t nabla[27];

    int idx_nabla = 0;
    double_t diffusionSum = 0;

    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {

          int currentidz = idz + dz;
          int currentidy = idy + dy;
          int currentidx = idx + dx;

          //checks for boundaries.
          if (currentidy >= 0 && currentidy < xsize && currentidx >= 0 && currentidx < ysize &&
              currentidz >= 0 && currentidz < zsize) {
            nabla[idx_nabla] =
                hostImage[currentidz * xsize * ysize + currentidy * ysize + currentidx] - center;

            double scaledDiff = pow(nabla[idx_nabla] / kappa, 2);

            if (diffusionOption == 1) {
              diffusionSum += nabla[idx_nabla] * exp(-scaledDiff);
            } else if (diffusionOption == 2) {
              diffusionSum += nabla[idx_nabla] / (1 + scaledDiff);
            } else {
              diffusionSum += nabla[idx_nabla] * (1 - tanh(scaledDiff));
            }
          } else {
            nabla[idx_nabla] = 0;
          }

          //update the index of nabla
          idx_nabla++;
        }
      }
    }

    outputImage[idx_center] = hostImage[idx_center] + deltaT * diffusionSum;
  }
}

template <typename dtype>
void anisotropicDiffusion3DGPU(dtype* hostImage, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize, int zsize) {
  dtype *deviceImage, *deviceTmp;
  size_t numBytes = xsize * ysize * zsize * sizeof(dtype);

  // Allocate memory for the input image on the device
  cudaMalloc((void**)&deviceImage, numBytes);
  cudaMalloc((void**)&deviceTmp, numBytes);

  cudaMemcpy(deviceImage, hostImage, numBytes, cudaMemcpyHostToDevice);

  dim3 blockSize(8, 8, 8);
  dim3 gridSize((ysize + blockSize.x - 1) / blockSize.x, (xsize + blockSize.y - 1) / blockSize.y,
                (zsize + blockSize.z - 1) / blockSize.z);

  for (int iter = 0; iter < totalIterations; iter++) {

    anisotropicDiffusion3DKernel<dtype><<<gridSize, blockSize>>>(
        deviceImage, deviceTmp, deltaT, kappa, diffusionOption, xsize, ysize, zsize);

    cudaDeviceSynchronize();  // Synchronous barrier at each time step iteration

    std::swap(deviceImage, deviceTmp);
  }

  cudaMemcpy(hostImage, deviceImage, numBytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceTmp);
}

template void anisotropicDiffusion3DGPU<float>(float*, int, float, float, int, int, int, int);
template void anisotropicDiffusion3DGPU<double>(double*, int, float, float, int, int, int, int);
