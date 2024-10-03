#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include <iostream>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/reconstruction_binary.h"

/**
 * @brief Perform recosntruction by erosion/dilation operation on the entire image using the GPU.
 * This function is meant to be called from host and slide the geodesic_morph_binary kernel function
 * through all pixels.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
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
void opening_by_reconstruction_on_device(dtype* hostImage, dtype* hostOutput, int* kernel,
                                         int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                         const int xsize, const int ysize, const int zsize,
                                         MorphOp operation, const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // set kenrel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  int hostFlagConverged = 1;

  // malloc device global memory
  dtype *deviceMarker, *deviceOutput, *deviceMask;
  int* deviceFlagConverged;
  int* deviceKernel;
  CHECK(cudaMalloc((dtype**)&deviceMarker, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceMask, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));
  CHECK(cudaMalloc((int**)&deviceFlagConverged, sizeof(int)));

  // transfer data from the host to the device
  CHECK(cudaMemset(deviceMarker, 0, nBytes));  //Initialize marker image to zero
  CHECK(cudaMemcpy(deviceMask, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceFlagConverged, &hostFlagConverged, sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // Create Marker from Erosion
  morph_binary(deviceMask, deviceMarker, xsize, ysize, zsize, flag_verbose, 0, 0, deviceKernel,
               kernel_xsize, kernel_ysize, kernel_zsize, EROSION);

  reconstruction_binary(deviceMarker, deviceMask, deviceOutput, xsize, ysize, zsize, operation,
                        flag_verbose);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free device memorys
  cudaFree(deviceMarker);
  cudaFree(deviceMask);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
template void opening_by_reconstruction_on_device<int>(int*, int*, int*, int, int, int, const int,
                                                       const int, const int, MorphOp, const int);
template void opening_by_reconstruction_on_device<unsigned int>(unsigned int*, unsigned int*, int*,
                                                                int, int, int, const int, const int,
                                                                const int, MorphOp, const int);
template void opening_by_reconstruction_on_device<uint16_t>(uint16_t*, uint16_t*, int*, int, int,
                                                            int, const int, const int, const int,
                                                            MorphOp, const int);
