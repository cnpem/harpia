#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/geodesic_morph_binary.h"

/**
 * @brief Perform geodesic erosion/dilation operation for one pixel.
 *
 * @tparam dtype The data type of the image.
 * @param image Input image (corresponds to the marker image).
 * @param output Output image.
 * @param mask Mask image.
 * @param centerIdx Center index in the x-dimension.
 * @param centerIdy Center index in the y-dimension.
 * @param centerIdz Center index in the z-dimension.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
CUDA_HOSTDEV void geodesic_morph_binary_pixel(dtype* image, dtype* mask, dtype* output,
                                              const int xsize, const int ysize, const int zsize,
                                              int centerIdx, int centerIdy, int centerIdz,
                                              int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                              MorphOp operation) {

  dtype* im = image;
  dtype kernel = 1;  //the kernel for geodesic operations is fixed in 1's.
  dtype aux;
  if (operation == EROSION) {
    aux = 1;  //erosion operation
  } else {
    aux = 0;  //dilation operation
  }

  int startIdx = centerIdx - kernel_xsize / 2;
  int startIdy = centerIdy - kernel_ysize / 2;
  int startIdz = centerIdz - kernel_zsize / 2;

  int imageIdx, imageIdy, imageIdz, index;

  //erosion/dilation operation
  for (int iz = 0; iz < kernel_zsize; iz++) {
    for (int iy = 0; iy < kernel_ysize; iy++) {
      for (int ix = 0; ix < kernel_xsize; ix++) {

        imageIdx = startIdx + ix;
        imageIdy = startIdy + iy;
        imageIdz = startIdz + iz;
        index = imageIdz * xsize * ysize + imageIdy * xsize + imageIdx;

        // ignore out of bounds pixels
        if (imageIdx < 0 || imageIdx > xsize - 1 || imageIdy < 0 || imageIdy > ysize - 1 ||
            imageIdz < 0 || imageIdz > zsize - 1) {
          // do nothing.
        }

        else {
          if (operation == EROSION) {
            aux = (im[index] == kernel) && aux;  //erosion operation
          } else {
            aux = (im[index] == kernel) || aux;  //dilation operation
          }
        }
      }
    }
  }

  int centerIndex = centerIdz * ysize * xsize + centerIdy * xsize + centerIdx;

  // intersection/union operation
  if (operation == EROSION) {
    output[centerIndex] = aux || mask[centerIndex];  //erosion operation
  } else {
    output[centerIndex] = aux && mask[centerIndex];  //dilation operation
  }
}
template CUDA_HOSTDEV void geodesic_morph_binary_pixel<int>(int*, int*, int*, const int, const int,
                                                            const int, int, int, int, int, int, int,
                                                            MorphOp);
template CUDA_HOSTDEV void geodesic_morph_binary_pixel<unsigned int>(unsigned int*, unsigned int*,
                                                                     unsigned int*, const int,
                                                                     const int, const int, int, int,
                                                                     int, int, int, int, MorphOp);
template CUDA_HOSTDEV void geodesic_morph_binary_pixel<uint16_t>(uint16_t*, uint16_t*, uint16_t*,
                                                                 const int, const int, const int,
                                                                 int, int, int, int, int, int,

                                                                 MorphOp);

/**
 * @brief Kernel function to perform geodesic erosion/dilation operation on the entire image.
 *
 * @tparam dtype The data type of the image.
 * @param deviceImage Input image on the device (corrresponds to the marker image).
 * @param deviceOutput Output image on the device.
 * @param deviceMask Mask image on the device.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
__global__ void geodesic_morph_binary_kernel(dtype* deviceImage, dtype* deviceMask,
                                             dtype* deviceOutput, const int xsize, const int ysize,
                                             const int zsize, int kernel_xsize, int kernel_ysize,
                                             int kernel_zsize, MorphOp operation) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    geodesic_morph_binary_pixel(deviceImage, deviceMask, deviceOutput, xsize, ysize, zsize, idx,
                                idy, idz, kernel_xsize, kernel_ysize, kernel_zsize, operation);
  }
}
template __global__ void geodesic_morph_binary_kernel<int>(int*, int*, int*, const int, const int,
                                                           const int, int, int, int, MorphOp);
template __global__ void geodesic_morph_binary_kernel<unsigned int>(unsigned int*, unsigned int*,
                                                                    unsigned int*, const int,
                                                                    const int, const int, int, int,
                                                                    int, MorphOp);
template __global__ void geodesic_morph_binary_kernel<uint16_t>(uint16_t*, uint16_t*, uint16_t*,
                                                                const int, const int, const int,
                                                                int, int, int, MorphOp);

template <typename dtype>
void geodesic_morph_binary(dtype* deviceImage, dtype* deviceMask, dtype* deviceOutput,
                           const int xsize, const int ysize, const int zsize,
                           const int flag_verbose, MorphOp operation) {
  //define connectivity kernel size for images of any dimension
  int kernel_xsize = (xsize > 2) ? 3 : xsize;
  int kernel_ysize = (ysize > 2) ? 3 : ysize;
  int kernel_zsize = (zsize > 2) ? 3 : zsize;

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
  geodesic_morph_binary_kernel<<<grid, block>>>(deviceImage, deviceMask, deviceOutput, xsize, ysize,
                                                zsize, kernel_xsize, kernel_ysize, kernel_zsize,
                                                operation);
  cudaDeviceSynchronize();  //assures all gpu threads are fineshed
}
template void geodesic_morph_binary<int>(int*, int*, int*, const int, const int, const int,
                                         const int, MorphOp);
template void geodesic_morph_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                  const int, const int, const int, const int,
                                                  MorphOp);
template void geodesic_morph_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int, const int,
                                              const int, const int, MorphOp);

/**
 * @brief Perform geodesic erosion/dilation operation on the entire image using the GPU. This
 * function is meant to be called from host and slide the morph_binary kerel function through all
 * pixels.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 */
template <typename dtype>
void geodesic_morph_binary_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                     const int xsize, const int ysize, const int zsize,
                                     const int flag_verbose, MorphOp operation) {
  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // malloc device global memory
  dtype *deviceImage, *deviceOutput, *deviceMask;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceMask, nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceMask, hostMask, nBytes, cudaMemcpyHostToDevice));

  // device erosion/dialation
  geodesic_morph_binary(deviceImage, deviceMask, deviceOutput, xsize, ysize, zsize, flag_verbose,
                        operation);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free host memorys
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
}
template void geodesic_morph_binary_on_device<int>(int*, int*, int*, const int, const int,
                                                   const int, const int, MorphOp);
template void geodesic_morph_binary_on_device<unsigned int>(unsigned int*, unsigned int*,
                                                            unsigned int*, const int, const int,
                                                            const int, const int, MorphOp);
template void geodesic_morph_binary_on_device<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int,
                                                        const int, const int, const int, MorphOp);

/**
 * @brief Perform geodesic erosion/dilation operation on the entire image using the CPU. This
 * function is used to check GPU results correctness.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 */
template <typename dtype>
void geodesic_morph_binary_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation) {

  //define connectivity kernel size for images of any dimension
  int kernel_xsize = (xsize > 2) ? 3 : xsize;
  int kernel_ysize = (ysize > 2) ? 3 : ysize;
  int kernel_zsize = (zsize > 2) ? 3 : zsize;

  for (int idz = 0; idz < zsize; idz++) {
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {

        geodesic_morph_binary_pixel(hostImage, hostMask, hostOutput, xsize, ysize, zsize, idx, idy,
                                    idz, kernel_xsize, kernel_ysize, kernel_zsize, operation);
      }
    }
  }  // slide over image
}
template void geodesic_morph_binary_on_host<int>(int*, int*, int*, const int, const int, const int,
                                                 MorphOp);
template void geodesic_morph_binary_on_host<unsigned int>(unsigned int*, unsigned int*,
                                                          unsigned int*, const int, const int,
                                                          const int, MorphOp);
template void geodesic_morph_binary_on_host<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int,
                                                      const int, const int, MorphOp);
