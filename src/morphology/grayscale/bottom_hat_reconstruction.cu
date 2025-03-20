#include <stdio.h>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/bottom_hat_reconstruction.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/reconstruction_grayscale.h"
#include "../../../include/morphology/subtraction.h"


template <typename dtype>
void bottom_hat_reconstruction_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                         const int ysize, const int zsize, const int flag_verbose,
                                         int* kernel, int kernel_xsize, int kernel_ysize,
                                         int kernel_zsize) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Set kernel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // Malloc device global memory
  dtype *deviceImage, *deviceTmp, *deviceAux;
  int* deviceKernel;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceAux, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // Closing operation: erosion followed by dilation
  morph_grayscale(deviceImage, deviceAux, xsize, ysize, zsize, flag_verbose, 0, 0, deviceKernel,
                  kernel_xsize, kernel_ysize, kernel_zsize, DILATION);

  morph_grayscale(deviceAux, deviceTmp, xsize, ysize, zsize, flag_verbose, 0, 0, deviceKernel,
                  kernel_xsize, kernel_ysize, kernel_zsize, EROSION);

  // Closing by reconstruction with closing as the marker image
  reconstruction_grayscale(deviceTmp, deviceImage, deviceAux, xsize, ysize, zsize, EROSION,
                           flag_verbose);

  // bottom-hat: closing - f
  subtraction(deviceImage, deviceAux, size, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceAux, nBytes, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(deviceTmp);
  cudaFree(deviceImage);
  cudaFree(deviceAux);
  cudaFree(deviceKernel);
}
// Template instantiations for specific types
template void bottom_hat_reconstruction_on_device<int>(int*, int*, const int, const int, const int,
                                                       const int, int*, int, int, int);
template void bottom_hat_reconstruction_on_device<unsigned int>(unsigned int*, unsigned int*,
                                                                const int, const int, const int,
                                                                const int, int*, int, int, int);
template void bottom_hat_reconstruction_on_device<float>(float*, float*, const int, const int,
                                                         const int, const int, int*, int, int, int);


template <typename dtype>
void bottom_hat_reconstruction_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                       const int ysize, const int zsize, int* kernel,
                                       int kernel_xsize, int kernel_ysize, int kernel_zsize) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Allocate temporary memory
  dtype* host_tmp = (dtype*)malloc(nBytes);

  // Opening operation
  MorphChain opening = {DILATION, EROSION};
  morph_chain_grayscale_on_host(hostImage, host_tmp, xsize, ysize, zsize, kernel, kernel_xsize,
                                kernel_ysize, kernel_zsize, opening);

  reconstruction_grayscale_on_host(host_tmp, hostImage, hostOutput, xsize, ysize, zsize, EROSION);

  // bottom-hat: closing - f
  subtraction_on_host(hostImage, hostOutput, size);

  // Free temporary memory
  free(host_tmp);
}
// Template instantiations for specific types
template void bottom_hat_reconstruction_on_host<int>(int*, int*, const int, const int, const int,
                                                     int*, int, int, int);
template void bottom_hat_reconstruction_on_host<unsigned int>(unsigned int*, unsigned int*,
                                                              const int, const int, const int, int*,
                                                              int, int, int);
template void bottom_hat_reconstruction_on_host<float>(float*, float*, const int, const int,
                                                       const int, int*, int, int, int);