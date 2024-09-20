#include <stdio.h>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_grayscale.h"
/**
 * @brief Performs grayscale morphological operation on a single pixel.
 * 
 * This function processes a single pixel in the grayscale image based on a given kernel and 
 * morphological operation (erosion or dilation). It considers the pixel's neighborhood defined by 
 * the kernel.
 * 
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param image Pointer to the input grayscale image.
 * @param output Pointer to the output image where the result will be stored.
 * @param centerIdx X-coordinate of the central pixel in the image.
 * @param centerIdy Y-coordinate of the central pixel in the image.
 * @param centerIdz Z-coordinate of the central pixel in the image.
 * @param kernel Pointer to the kernel used for the morphological operation.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation The morphological operation to perform (erosion or dilation).
 */
template <typename dtype>
CUDA_HOSTDEV void morph_grayscale_pixel(dtype* image, dtype* output, const int xsize,
                                        const int ysize, const int zsize, int centerIdx,
                                        int centerIdy, int centerIdz, int* kernel, int kernel_xsize,
                                        int kernel_ysize, int kernel_zsize, MorphOp operation) {
  dtype* im = image;
  int* ik = kernel;

  // Initialize auxiliary value with the central pixel
  dtype aux = im[centerIdz * xsize * ysize + centerIdy * xsize + centerIdx];

  int imageIdx, imageIdy, imageIdz, index;

  int startIdx = centerIdx - kernel_xsize / 2;
  int startIdy = centerIdy - kernel_ysize / 2;
  int startIdz = centerIdz - kernel_zsize / 2;

  for (int iz = 0; iz < kernel_zsize; iz++) {
    for (int iy = 0; iy < kernel_ysize; iy++) {
      for (int ix = 0; ix < kernel_xsize; ix++) {

        imageIdx = startIdx + ix;
        imageIdy = startIdy + iy;
        imageIdz = startIdz + iz;
        index = imageIdz * xsize * ysize + imageIdy * xsize + imageIdx;

        // Ignore out of bounds pixels and don't care pixels
        if (imageIdx < 0 || imageIdx > xsize - 1 || imageIdy < 0 || imageIdy > ysize - 1 ||
            imageIdz < 0 || imageIdz > zsize - 1 || ik[ix] < 0) {
          // Do nothing.
        } else {
          if (operation == EROSION) {
            aux = (im[index] < aux) ? im[index] : aux;  // Erosion: aux is the min value
          } else {
            aux = (im[index] > aux) ? im[index] : aux;  // Dilation: aux is the max value
          }
        }
      }
    }
  }
  output[centerIdz * ysize * xsize + centerIdy * xsize + centerIdx] = aux;
}
template CUDA_HOSTDEV void morph_grayscale_pixel<unsigned int>(unsigned int*, unsigned int*,
                                                               const int, const int, const int, int,
                                                               int, int, int*, int, int, int,

                                                               MorphOp);
template CUDA_HOSTDEV void morph_grayscale_pixel<int>(int*, int*, const int, const int, const int,
                                                      int, int, int, int*, int, int, int, MorphOp);
template CUDA_HOSTDEV void morph_grayscale_pixel<float>(float*, float*, const int, const int,
                                                        const int, int, int, int, int*, int, int,
                                                        int, MorphOp);

template <typename dtype>
__global__ void morph_grayscale_kernel(dtype* deviceImage, dtype* deviceOutput, const int xsize,
                                       const int ysize, const int zsize, int* kernel,
                                       int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                       MorphOp operation) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    morph_grayscale_pixel(deviceImage, deviceOutput, xsize, ysize, zsize, idx, idy, idz, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, operation);
  }
}
template __global__ void morph_grayscale_kernel<unsigned int>(unsigned int*, unsigned int*,
                                                              const int, const int, const int, int*,
                                                              int, int, int, MorphOp);
template __global__ void morph_grayscale_kernel<int>(int*, int*, const int, const int, const int,
                                                     int*, int, int, int, MorphOp);
template __global__ void morph_grayscale_kernel<float>(float*, float*, const int, const int,
                                                       const int, int*, int, int, int, MorphOp);

template <typename dtype>
void morph_grayscale(dtype* deviceImage, dtype* deviceOutput, const int xsize, const int ysize,
                     const int zsize, int* deviceKernel, int kernel_xsize, int kernel_ysize,
                     int kernel_zsize, MorphOp operation, const int flag_verbose) {
  // Set up execution configuration
  dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
  if (zsize == 1)
    block = dim3(BLOCK_2D, BLOCK_2D, 1);
  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  // Check grid and block dimensions from host side
  if (flag_verbose) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  // Device erosion/dilation
  morph_grayscale_kernel<<<grid, block>>>(deviceImage, deviceOutput, xsize, ysize, zsize,
                                          deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
                                          operation);
  cudaDeviceSynchronize();  // Assures all GPU threads are finished
}
template void morph_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                            const int, int*, int, int, int, MorphOp, const int);
template void morph_grayscale<int>(int*, int*, const int, const int, const int, int*, int, int, int,
                                   MorphOp, const int);
template void morph_grayscale<float>(float*, float*, const int, const int, const int, int*, int,
                                     int, int, MorphOp, const int);

/**
 * @brief Applies grayscale morphological operations on the device.
 * 
 * This function sets up the execution configuration and memory on the device, launches the CUDA 
 * kernel to apply the morphological operation across the image, and then transfers the result back 
 * to the host.
 * 
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param hostImage Pointer to the input grayscale image on the host.
 * @param hostOutput Pointer to the output image where the result will be stored on the host.
 * @param kernel Pointer to the kernel used for the morphological operation.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation The morphological operation to perform (erosion or dilation).
 * @param flag_verbose If non-zero, print verbose output about the grid and block dimensions.
 */
template <typename dtype>
void morph_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, int* kernel, int kernel_xsize,
                               int kernel_ysize, int kernel_zsize, MorphOp operation,
                               const int flag_verbose) {
  // Set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Set kernel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // Allocate device global memory
  dtype *deviceImage, *deviceOutput;
  int* deviceKernel;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // Device erosion/dilation
  morph_grayscale(deviceImage, deviceOutput, xsize, ysize, zsize, deviceKernel, kernel_xsize,
                  kernel_ysize, kernel_zsize, operation, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
template void morph_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                      const int, const int, int*, int, int, int,
                                                      MorphOp, const int);
template void morph_grayscale_on_device<int>(int*, int*, const int, const int, const int, int*, int,
                                             int, int, MorphOp, const int);
template void morph_grayscale_on_device<float>(float*, float*, const int, const int, const int,
                                               int*, int, int, int, MorphOp, const int);

/**
 * @brief Applies grayscale morphological operations on the host.
 * 
 * This function iterates over all pixels in the image and applies the morphological operation 
 * using the `morph_grayscale_pixel` function on the CPU.
 * 
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param hostImage Pointer to the input grayscale image on the host.
 * @param hostOutput Pointer to the output image where the result will be stored on the host.
 * @param kernel Pointer to the kernel used for the morphological operation.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation The morphological operation to perform (erosion or dilation).
 */
template <typename dtype>
void morph_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize, MorphOp operation) {
  for (int idz = 0; idz < zsize; idz++) {
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {

        morph_grayscale_pixel(hostImage, hostOutput, xsize, ysize, zsize, idx, idy, idz, kernel,
                              kernel_xsize, kernel_ysize, kernel_zsize, operation);
      }
    }
  }  // Slide over image
}
template void morph_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                    const int, const int, int*, int, int, int,
                                                    MorphOp);
template void morph_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*, int,
                                           int, int, MorphOp);
template void morph_grayscale_on_host<float>(float*, float*, const int, const int, const int, int*,
                                             int, int, int, MorphOp);