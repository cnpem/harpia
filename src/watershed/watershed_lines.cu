#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/watershed/watershed_lines.h"

//since we shall calculate this for visualization purpose, we dont need to compute the boundaries for the whole image, just the current slice
//so consider a single slice of labels generated from watershed as input (it does not need to be necessarily from watershed, but we will use for it only)
template <typename in_dtype>
__device__ void isBoundary2d(in_dtype* labels,int* boundaries, int idx, int idy, int xsize, int ysize)
{

    int index = idy * xsize + idx;

    //borders wont be considered as a boundary
    if (idy - 1 < 0 || idx - 1 < 0 || idy + 1 >= ysize || idx + 1 >= xsize)
    {
        return;
    }

    // Define the dimensions of the kernel
  const int nx = 3;
  const int ny = 3;

  int inputY;
  int inputX;

  // Iterate over the 9 neighbors

    for (int m = 0; m < nx; ++m)
    {

        for (int n = 0; n < ny; ++n) 
        {

            // Compute the position with respect to the center of the kernel
            inputY = idy - nx / 2 + m;
            inputX = idx - ny / 2 + n;

            // Skip the center voxel itself
            if (m == nx / 2 && n == ny / 2) 
            {
                continue;
            }

            // Check for boundary conditions
            if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize) 
            {
                
                //there wont be any competition between threads, since each thread operates over one pixel.
                if (labels[index] != labels[inputY * xsize + inputX]) 
                {
                    boundaries[index] = 1;
                    return;
                }
            }
        }
    }
      
}

template <typename in_dtype>
__global__ void boundaries2d(in_dtype* labels,int* boundaries, int xsize, int ysize)
{

  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int idy = (threadIdx.y + blockIdx.y * blockDim.y);

  if (idx < xsize && idy < ysize)
  {
    isBoundary2d(labels, boundaries, idx, idy, xsize, ysize);
  }

}

// Explicit template instantiations
template __global__ void boundaries2d<int>(int* labels, int* boundaries, int xsize, int ysize);
template __global__ void boundaries2d<unsigned int>(unsigned int* labels, int* boundaries, int xsize, int ysize);

template<typename in_dtype>
void compute_boundaries2d(in_dtype* labels,int* boundaries, int xsize, int ysize) {
    int* deviceLabels;
    int* deviceBoundaries;

    cudaMalloc(&deviceLabels, xsize * ysize * sizeof(in_dtype));
    cudaMalloc(&deviceBoundaries, xsize * ysize * sizeof(unsigned int));

    cudaMemcpy(deviceLabels, labels, xsize * ysize  * sizeof(in_dtype), cudaMemcpyHostToDevice);
    cudaMemset(deviceBoundaries, 0, xsize * ysize * sizeof(int));

    dim3 blockDim(32, 32);
    dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y);

    boundaries2d<<<gridDim,blockDim>>>(deviceLabels,deviceBoundaries,xsize,ysize);

    cudaMemcpy(boundaries, deviceBoundaries, xsize * ysize * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(deviceLabels);
    cudaFree(deviceBoundaries);
}




template void compute_boundaries2d<int>(int* labels, int* boundaries, int xsize, int ysize);
template void compute_boundaries2d<unsigned int>(unsigned int* labels, int* boundaries, int xsize, int ysize);