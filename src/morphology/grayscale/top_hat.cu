#include <stdio.h>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/subtraction.h"
#include "../../../include/morphology/top_hat.h"

template <typename dtype>
void top_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, const int padding_bottom,
                       const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                       int kernel_zsize) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);
  size_t nBytes_padding = xsize * ysize * (padding_bottom + padding_top) * sizeof(dtype);

  int half_padding_bottom = padding_bottom / 2;
  int half_padding_top = padding_top / 2;
  size_t nBytes_half_padding =
      xsize * ysize * (half_padding_bottom + half_padding_top) * sizeof(dtype);

  size_t nBytes_input = nBytes + nBytes_padding;
  size_t nBytes_tmp = nBytes + nBytes_half_padding;

  // set kernel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // malloc device global memory
  dtype *ii_hostImage, *ii_deviceImage, *i_deviceImage, *i_deviceAux, *deviceImage, *deviceAux,
      *deviceTmp;
  int* deviceKernel;

  CHECK(cudaMalloc((dtype**)&ii_deviceImage, nBytes_input));
  CHECK(cudaMalloc((dtype**)&i_deviceAux, nBytes_tmp));
  CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // transfer input + padding
  ii_hostImage = hostImage - padding_bottom * xsize * ysize;

  CHECK(cudaMemcpy(ii_deviceImage, ii_hostImage, nBytes_input, cudaMemcpyHostToDevice));

  i_deviceImage = ii_deviceImage + half_padding_bottom * xsize * ysize;

  // Perform the first operation in the chain
  morph_grayscale(i_deviceImage, i_deviceAux, xsize, ysize,
                  zsize + half_padding_top + half_padding_bottom, flag_verbose, half_padding_bottom,
                  half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
                  EROSION);

  deviceAux = i_deviceAux + half_padding_bottom * xsize * ysize;

  // Perform the second operation in the chain
  morph_grayscale(deviceAux, deviceTmp, xsize, ysize, zsize, flag_verbose, half_padding_bottom,
                  half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
                  DILATION);

  deviceImage = i_deviceImage + half_padding_bottom * xsize * ysize;

  // Top-hat: input - opening
  subtraction(deviceTmp, deviceImage, size, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceImage, nBytes, cudaMemcpyDeviceToHost));

  // free device memory
  cudaFree(ii_deviceImage);
  cudaFree(i_deviceAux);
  cudaFree(deviceTmp);
  cudaFree(deviceKernel);
}
// Template instantiations for specific types
template void top_hat_on_device<int>(int*, int*, const int, const int, const int, const int,
                                     const int, const int, int*, int, int, int);
template void top_hat_on_device<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, const int, const int, int*, int,
                                              int, int);
template void top_hat_on_device<float>(float*, float*, const int, const int, const int, const int,
                                       const int, const int, int*, int, int, int);

template <typename dtype>
void top_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                     const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                     int kernel_zsize) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Allocate temporary memory
  dtype* host_tmp = (dtype*)malloc(nBytes);

  // Opening operation
  MorphChain opening = {EROSION, DILATION};
  morph_chain_grayscale_on_host(hostImage, host_tmp, xsize, ysize, zsize, kernel, kernel_xsize,
                                kernel_ysize, kernel_zsize, opening);

  // Top-hat: f - opening
  memcpy(hostOutput, hostImage, nBytes);
  subtraction_on_host(host_tmp, hostOutput, size);

  // Free temporary memory
  free(host_tmp);
}
// Template instantiations for specific types
template void top_hat_on_host<int>(int*, int*, const int, const int, const int, int*, int, int,
                                   int);
template void top_hat_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                            const int, int*, int, int, int);
template void top_hat_on_host<float>(float*, float*, const int, const int, const int, int*, int,
                                     int, int);