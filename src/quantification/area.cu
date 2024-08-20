#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/quantification/area.h"

__device__ void isArea(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize,
                       int ysize, int zsize) {
  int imageIndex = idz * xsize * ysize + idy * xsize + idx;
  int counterIndex = image[imageIndex];

  atomicAdd(&counter[counterIndex], 1);
}

__device__ void isSurface(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize,
                          int ysize, int zsize) {
  int imageIndex = idz * xsize * ysize + idy * xsize + idx;
  int counterIndex = image[imageIndex];

  // All borders are perimeters (this is sufficient for this kernel format).
  if (idy - 1 < 0 || idx - 1 < 0 || idz - 1 < 0 || idy + 1 >= ysize || idx + 1 >= xsize ||
      idz + 1 >= zsize) {
    atomicAdd(&counter[counterIndex], 1);
    return;
  }

  // Define the dimensions of the kernel
  const int nz = 3;
  const int nx = 3;
  const int ny = 3;

  int inputZ;
  int inputY;
  int inputX;

  // Iterate over the 26 neighbors
  for (int l = 0; l < nz; ++l) {

    for (int m = 0; m < nx; ++m) {

      for (int n = 0; n < ny; ++n) {

        // Compute the position with respect to the center of the kernel
        inputZ = idz - nz / 2 + l;
        inputY = idy - nx / 2 + m;
        inputX = idx - ny / 2 + n;

        // Skip the center voxel itself
        if (l == nz / 2 && m == nx / 2 && n == ny / 2) {
          continue;
        }

        // Check for boundary conditions
        if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize && inputZ >= 0 &&
            inputZ < zsize) {

          if (image[imageIndex] != image[inputZ * xsize * ysize + inputY * xsize + inputX]) {
            atomicAdd(&counter[counterIndex], 1);
            return;
          }
        }
      }
    }
  }
}

__global__ void area_counter(int* image, unsigned int* counter, int idz, int xsize, int ysize,
                             int zsize) {
  // To compute the area, we just need to make the accumulated sum.
  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int idy = (threadIdx.y + blockIdx.y * blockDim.y);

  if (idx < xsize && idy < ysize) {
    isArea(image, counter, idx, idy, idz, xsize, ysize, zsize);
  }
}

__global__ void surface_area_counter(int* image, unsigned int* counter, int idz, int xsize,
                                     int ysize, int zsize) {

  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int idy = (threadIdx.y + blockIdx.y * blockDim.y);

  if (idx < xsize && idy < ysize) {
    isSurface(image, counter, idx, idy, idz, xsize, ysize, zsize);
  }
}

void area(int* image, unsigned int* output, int xsize, int ysize, int zsize, bool type) {
  int* deviceImage;
  unsigned int* deviceOutput;

  cudaMalloc(&deviceImage, xsize * ysize * zsize * sizeof(int));
  cudaMalloc(&deviceOutput, xsize * ysize * zsize * sizeof(unsigned int));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(deviceOutput, 0,
             xsize * ysize * zsize * sizeof(unsigned int));  // Initialize output array to zero

  dim3 blockDim(32, 32);
  dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y);

  if (type == false) {
    // Computes the area of a 2D object in each slice.
    for (int idz = 0; idz < zsize; idz++) {
      area_counter<<<gridDim, blockDim>>>(deviceImage, deviceOutput, idz, xsize, ysize, zsize);
    }
  }

  else {
    // Computes the surface area of a 3D object.

    for (int idz = 0; idz < zsize; idz++) {
      surface_area_counter<<<gridDim, blockDim>>>(deviceImage, deviceOutput, idz, xsize, ysize,
                                                  zsize);
    }
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

    unsigned int hostOutput[xsize * ysize * zsize] = {0};

    int image[xsize * ysize * zsize] = 
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

    area(image, hostOutput, xsize, ysize, zsize, 1);

    for (int i = 0; i < ysize; i++) {
        for (int j = 0; j < xsize; j++) {
            std::cout << hostOutput[i*xsize +j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/