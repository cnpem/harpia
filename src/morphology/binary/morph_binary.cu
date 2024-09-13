#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_binary.h"

/**
 * @brief Perform erosion/dilation operation for one pixel.
 * 
 * @tparam dtype The data type of the image.
 * @param image Input image.
 * @param output Output image.
 * @param centerIdx Center index in the x-dimension.
 * @param centerIdy Center index in the y-dimension.
 * @param centerIdz Center index in the z-dimension.
 * @param kernel Morphological operation kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
CUDA_HOSTDEV void morph_binary_pixel(dtype* image, dtype* output, const int xsize, const int ysize,
                                     const int zsize, int centerIdx, int centerIdy, int centerIdz,
                                     int* kernel, int kernel_xsize, int kernel_ysize,
                                     int kernel_zsize, MorphOp operation) {
  dtype* im = image;
  int* ik = kernel;
  dtype aux;
  if (operation == EROSION) {
    aux = 1;  //erosion operation
  } else {
    aux = 0;  //dilation operation
  }

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

        // ignore out of bounds pixels and don't care pixels
        // don't care pixels are signaled as -1 in the kernel
        if (imageIdx < 0 || imageIdx > xsize - 1 || imageIdy < 0 || imageIdy > ysize - 1 ||
            imageIdz < 0 || imageIdz > zsize - 1 || ik[ix] < 0) {
          // do nothing.
        }

        else {
          if (operation == EROSION) {
            aux = (im[index] == (dtype)ik[ix]) && aux;  //erosion operation
          } else {
            aux = (im[index] == (dtype)ik[ix]) || aux;  //dilation operation
          }
        }
      }
      ik += kernel_xsize;
    }
  }
  output[centerIdz * ysize * xsize + centerIdy * xsize + centerIdx] = aux;
}
template CUDA_HOSTDEV void morph_binary_pixel<int>(int*, int*, const int, const int, const int, int,
                                                   int, int, int*, int, int, int, MorphOp);
template CUDA_HOSTDEV void morph_binary_pixel<unsigned int>(unsigned int*, unsigned int*, const int,
                                                            const int, const int, int, int, int,
                                                            int*, int, int, int, MorphOp);
template CUDA_HOSTDEV void morph_binary_pixel<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                        const int, int, int, int, int*, int, int,
                                                        int, MorphOp);

/**
 * @brief Kernel function to perform erosion/dilation operation on the entire image.
 * 
 * @tparam dtype The data type of the image.
 * @param deviceImage Input image on the device.
 * @param deviceOutput Output image on the device.
 * @param kernel Morphological operation kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
__global__ void morph_binary_kernel(dtype* deviceImage, dtype* deviceOutput, const int xsize,
                                    const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                    int kernel_ysize, int kernel_zsize, MorphOp operation) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    morph_binary_pixel(deviceImage, deviceOutput, xsize, ysize, zsize, idx, idy, idz, kernel,
                       kernel_xsize, kernel_ysize, kernel_zsize, operation);
  }
}
template __global__ void morph_binary_kernel<int>(int*, int*, const int, const int, const int, int*,
                                                  int, int, int, MorphOp);
template __global__ void morph_binary_kernel<unsigned int>(unsigned int*, unsigned int*, const int,
                                                           const int, const int, int*, int, int,
                                                           int, MorphOp);
template __global__ void morph_binary_kernel<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                       const int, int*, int, int, int, MorphOp);

template <typename dtype>
void morph_binary(dtype* deviceImage, dtype* deviceOutput, const int xsize, const int ysize,
                  const int zsize, int* deviceKernel, int kernel_xsize, int kernel_ysize,
                  int kernel_zsize, MorphOp operation, const int flag_verbose) {
  //set up execution configuratio
  dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
  if (zsize == 1)
    block = dim3(BLOCK_2D, BLOCK_2D, 1);

  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  // check grid and block dimension from host side
  if (flag_verbose) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  // device erosion/dialation
  morph_binary_kernel<<<grid, block>>>(deviceImage, deviceOutput, xsize, ysize, zsize, deviceKernel,
                                       kernel_xsize, kernel_ysize, kernel_zsize, operation);
  cudaDeviceSynchronize();  //assures all gpu threads are fineshed
}
template void morph_binary<int>(int*, int*, const int, const int, const int, int*, int, int, int,
                                MorphOp, const int);
template void morph_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                         const int, int*, int, int, int, MorphOp, const int);
template void morph_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int, int*,
                                     int, int, int, MorphOp, const int);

/**
 * @brief Perform erosion/dilation operation on the entire image using the GPU. This function is 
 * meant to be called from host and slide the morph_binary kerel function through all pixels.
 * 
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host.
 * @param hostOutput Output image on the host.
 * @param kernel Morphological operation kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 */
template <typename dtype>
void morph_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                            int kernel_zsize, MorphOp operation, const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // set kenrel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // malloc device global memory
  dtype *deviceImage, *deviceOutput;
  int* deviceKernel;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // device erosion/dialation
  morph_binary(deviceImage, deviceOutput, xsize, ysize, zsize, deviceKernel, kernel_xsize,
               kernel_ysize, kernel_zsize, operation, flag_verbose);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free host memorys
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
template void morph_binary_on_device<int>(int*, int*, const int, const int, const int, int*, int,
                                          int, int, MorphOp, const int);
template void morph_binary_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                   const int, const int, int*, int, int, int,
                                                   MorphOp, const int);
template void morph_binary_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                               const int, int*, int, int, int, MorphOp, const int);

/**
 * @brief Perform erosion/dilation operation on the entire image using the CPU. This function is 
 * used to check GPU results correctness.
 * 
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host.
 * @param hostOutput Output image on the host.
 * @param kernel Morphological operation kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
void morph_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                          int kernel_zsize, MorphOp operation) {

  for (int idz = 0; idz < zsize; idz++) {
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {

        morph_binary_pixel(hostImage, hostOutput, xsize, ysize, zsize, idx, idy, idz, kernel,
                           kernel_xsize, kernel_ysize, kernel_zsize, operation);
      }
    }
  }  // slide over image
}
template void morph_binary_on_host<int>(int*, int*, const int, const int, const int, int*, int, int,
                                        int, MorphOp);
template void morph_binary_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                                 const int, int*, int, int, int, MorphOp);
template void morph_binary_on_host<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                             int*, int, int, int, MorphOp);
