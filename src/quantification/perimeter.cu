#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/quantification/perimeter.h"

__device__ void isPerimeter(int* image, u_int* counter, int idx, int idy, int idz, int xsize,
                            int ysize, int zsize) {
  int imageIndex = idz * xsize * ysize + idy * xsize + idx;
  int counterIndex = image[imageIndex];

  // All borders are perimeters (this is sufficient for this kernel format).
  if (idx == 0 || idx == xsize - 1 || idy == 0 || idy == ysize - 1) {
    atomicAdd(&counter[counterIndex], 1);
    return;
  }

  // Define the 8 neighbors in a given slice
  int p1 = idz * xsize * ysize + (idy - 1) * xsize + idx;
  int p2 = idz * xsize * ysize + (idy - 1) * xsize + (idx - 1);
  int p3 = idz * xsize * ysize + idy * xsize + (idx - 1);
  int p4 = idz * xsize * ysize + (idy + 1) * xsize + (idx - 1);
  int p5 = idz * xsize * ysize + (idy + 1) * xsize + idx;
  int p6 = idz * xsize * ysize + (idy + 1) * xsize + (idx + 1);
  int p7 = idz * xsize * ysize + idy * xsize + (idx + 1);
  int p8 = idz * xsize * ysize + (idy - 1) * xsize + (idx + 1);

  // Only one case where the pixel is not a perimeter, for the given kernel.
  if (image[imageIndex] == image[p1] && image[imageIndex] == image[p2] &&
      image[imageIndex] == image[p3] && image[imageIndex] == image[p4] &&
      image[imageIndex] == image[p5] && image[imageIndex] == image[p6] &&
      image[imageIndex] == image[p7] && image[imageIndex] == image[p8]) {
    return;
  }

  atomicAdd(&counter[counterIndex], 1);
}

__global__ void perimeter_counter(int* image, u_int* counter, int xsize, int ysize, int zsize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    isPerimeter(image, counter, idx, idy, idz, xsize, ysize, zsize);
  }
}

void perimeter(int* image, u_int* output, int xsize, int ysize, int zsize) {
  int* deviceImage;
  u_int* deviceOutput;

  cudaMalloc(&deviceImage, xsize * ysize * zsize * sizeof(int));
  cudaMalloc(&deviceOutput, xsize * ysize * zsize * sizeof(u_int));

  cudaMemcpy(deviceImage, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(deviceOutput, 0,
             xsize * ysize * zsize * sizeof(u_int));  // Initialize output array to zero

  dim3 blockDim(8, 8, 8);  // Example block dimensions, can be adjusted
  dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y,
               (zsize + blockDim.z - 1) / blockDim.z);

  perimeter_counter<<<gridDim, blockDim>>>(deviceImage, deviceOutput, xsize, ysize, zsize);

  cudaMemcpy(output, deviceOutput, xsize * ysize * zsize * sizeof(u_int), cudaMemcpyDeviceToHost);

  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}

/*
int main()
{

    const int xsize = 8;
    const int ysize = 8;
    const int zsize = 2;

    u_int output[xsize*ysize*zsize];

    int hostImage[xsize*ysize*zsize] = 
    {
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 2, 2, 2, 0, 0,
         0, 0, 2, 2, 2, 2, 0, 0,
         0, 0, 0, 2, 2, 2, 0, 0,

         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 2, 2, 2, 0, 0,
         0, 0, 2, 2, 2, 2, 0, 0,
         0, 0, 0, 2, 2, 2, 0, 0
    };

    perimeter(hostImage,output,xsize,ysize,zsize);

    for (int i = 0; i < xsize*ysize*zsize; i++)
    {

        std::cout<<output[i]<<"\n";

    }

    return 0;


}
*/