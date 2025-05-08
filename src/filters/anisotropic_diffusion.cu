#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include "../../include/filters/anisotropic_diffusion.h"
#include "../../include/common/chunkedExecutor.h"
#include "../../include/common/grid_block_sizes.h"
#include "../../include/morphology/cuda_helper.h"

template <typename dtype>
__global__ void anisotropicDiffusion2DKernel(dtype* deviceImage, dtype* outputImage, float deltaT,
                                             float kappa, int diffusionOption, int xsize,
                                             int ysize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // x-axis
  int idy = blockIdx.y * blockDim.y + threadIdx.y;  // y-axis

  if (idx < xsize && idy < ysize) {  // Check within bounds for ysize (y-axis) and xsize (x-axis)

    // Compute indices for the neighboring cells with boundary checks
    int idyNorth = min(idy + 1, ysize - 1);
    int idySouth = max(idy - 1, 0);
    int idxEast = min(idx + 1, xsize - 1);
    int idxWest = max(idx - 1, 0);

    dtype center = deviceImage[idy * xsize + idx];
    dtype nabla[8];
    double_t diffusionCoefficients[8];

    nabla[0] = deviceImage[idyNorth * xsize + idx] - center;      // North
    nabla[1] = deviceImage[idySouth * xsize + idx] - center;      // South
    nabla[2] = deviceImage[idy * xsize + idxWest] - center;       // West
    nabla[3] = deviceImage[idy * xsize + idxEast] - center;       // East
    nabla[4] = deviceImage[idyNorth * xsize + idxWest] - center;  // Northwest
    nabla[5] = deviceImage[idyNorth * xsize + idxEast] - center;  // Northeast
    nabla[6] = deviceImage[idySouth * xsize + idxWest] - center;  // Southwest
    nabla[7] = deviceImage[idySouth * xsize + idxEast] - center;  // Southeast

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

    outputImage[idy * xsize + idx] = deviceImage[idy * xsize + idx] + deltaT * diffusionSum;
  }
}

template <typename dtype>
void anisotropicDiffusion2DGPU(dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize) {
  dtype *deviceImage, *deviceTmp;
  size_t numBytes = xsize * ysize * sizeof(dtype);

  // Allocate memory for the input image on the device
  cudaMalloc((void**)&deviceImage, numBytes);
  cudaMalloc((void**)&deviceTmp, numBytes);

  cudaMemcpy(deviceImage, hostImage, numBytes, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

  for (int iter = 0; iter < totalIterations; iter++) {
    anisotropicDiffusion2DKernel<dtype><<<gridSize, blockSize>>>(
        deviceImage, deviceTmp, deltaT, kappa, diffusionOption, xsize, ysize);

    cudaDeviceSynchronize();  // Synchronous barrier at each time step iteration

    std::swap(deviceImage, deviceTmp);
  }

  cudaMemcpy(hostOutput, deviceImage, numBytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceTmp);
}


template void anisotropicDiffusion2DGPU<float>(float*, float*, int, float, float, int, int, int);
template void anisotropicDiffusion2DGPU<double>(double*, double*, int, float, float, int, int, int);

// on device, change name
template <typename dtype>
__global__ void anisotropicDiffusion3DKernel(dtype* deviceImage, dtype* outputImage, float deltaT,
                                             float kappa, int diffusionOption,
                                             int xsize, int ysize, int zsize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;


    if (idx < xsize && idy < ysize && idz < zsize) {
        int idx_center = idx + idy * xsize + idz * xsize * ysize;
        dtype center = deviceImage[idx_center];
        double_t diffusionSum = 0;

        // Process 6-neighborhood (face neighbors)
        int offsets[6][3] = {{0,0,1}, {0,0,-1}, {0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0}};
        
        for (int i = 0; i < 6; i++) {
            int currentidz = idz + offsets[i][0];
            int currentidy = idy + offsets[i][1];
            int currentidx = idx + offsets[i][2];

            if (currentidx >= 0 && currentidx < xsize && 
                currentidy >= 0 && currentidy < ysize && 
                currentidz >= 0 && currentidz < zsize) {
                
                dtype nabla = deviceImage[currentidx + currentidy * xsize + 
                                      currentidz * xsize * ysize] - center;
                
                double scaledDiff = pow(nabla / kappa, 2);
                
                if (diffusionOption == 1) {
                    diffusionSum += nabla * exp(-scaledDiff);
                } else if (diffusionOption == 2) {
                    diffusionSum += nabla / (1 + scaledDiff);
                } else {
                    diffusionSum += nabla * (1 - tanh(scaledDiff));
                }
            }
        }

        outputImage[idx_center] = center + deltaT * diffusionSum;
    }
}

template <typename dtype>
void anisotropicDiffusion3DGPU(dtype* hostImage, dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                               int totalIterations, float deltaT, float kappa, int diffusionOption) {

  size_t numBytes = xsize * ysize * zsize * sizeof(dtype);

  // Allocate memory for the input image on the device
  dtype *deviceImage, *deviceTmp;
  CHECK(cudaMalloc((void**)&deviceImage, numBytes));
  CHECK(cudaMalloc((void**)&deviceTmp, numBytes));

  CHECK(cudaMemcpy(deviceImage, hostImage, numBytes, cudaMemcpyHostToDevice));
  //set up execution configuration
  dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
  if (zsize == 1)
    block = dim3(BLOCK_2D, BLOCK_2D, 1);
  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);
  // check grid and block dimension from host side
  if (flag_verbose==1) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  for (int iter = 0; iter < totalIterations; iter++) {
    anisotropicDiffusion3DKernel<dtype><<<grid, block>>>(
        deviceImage, deviceTmp, deltaT, kappa, diffusionOption, xsize, ysize, zsize);

    cudaDeviceSynchronize();  // Synchronous barrier at each time step iteration

    std::swap(deviceImage, deviceTmp);
  }

  CHECK(cudaMemcpy(hostOutput, deviceImage, numBytes, cudaMemcpyDeviceToHost));

  cudaFree(deviceImage);
  cudaFree(deviceTmp);
}

template void anisotropicDiffusion3DGPU<float>(float* , float*    , int, int, int, int, int, float, float, int);
template void anisotropicDiffusion3DGPU<double>(double* , double* , int, int, int, int, int, float, float, int);

template <typename dtype>
void anisotropicDiffusion3D(dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize, int zsize, const int flag_verbose, float gpuMemory, int ngpus) {
    if (ngpus == 0) {
      throw std::runtime_error("CPU implementation is not available for anisotropicDiffusion3D. "
        "Please ensure a GPU is available to execute this function.");

    } else {
      int ncopies = 2;
      chunkedExecutor(anisotropicDiffusion3DGPU<dtype>, ncopies, gpuMemory, ngpus, hostImage,
        hostOutput, xsize, ysize, zsize, flag_verbose, totalIterations, deltaT, kappa, diffusionOption);
  }
}

template void anisotropicDiffusion3D<float>(float*, float*,    int, float, float, int, int, int, int, int, float, int);
template void anisotropicDiffusion3D<double>(double*, double*, int, float, float, int, int, int, int, int, float, int);

