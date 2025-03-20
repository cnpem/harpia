#include <stdio.h>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/bottom_hat.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/subtraction.h"

template <typename dtype>
void bottom_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, const int flag_verbose, const int padding_bottom,
                          const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                          int kernel_zsize) {
  // set input dimension
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
  dtype *ii_hostImage, *ii_deviceImage, *i_deviceImage, *i_deviceTmp, *deviceImage, *deviceTmp,
      *deviceOutput;
  int* deviceKernel;

  CHECK(cudaMalloc((dtype**)&ii_deviceImage, nBytes_input));
  CHECK(cudaMalloc((dtype**)&i_deviceTmp, nBytes_tmp));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // transfer input + padding
  ii_hostImage = hostImage - padding_bottom * xsize * ysize;

  CHECK(cudaMemcpy(ii_deviceImage, ii_hostImage, nBytes_input, cudaMemcpyHostToDevice));

  i_deviceImage = ii_deviceImage + half_padding_bottom * xsize * ysize;

  // Perform the first operation in the chain
  morph_grayscale(i_deviceImage, i_deviceTmp, xsize, ysize,
                  zsize + half_padding_top + half_padding_bottom, flag_verbose, half_padding_bottom,
                  half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
                  DILATION);

  deviceTmp = i_deviceTmp + half_padding_bottom * xsize * ysize;

  // Perform the second operation in the chain
  morph_grayscale(deviceTmp, deviceOutput, xsize, ysize, zsize, flag_verbose, half_padding_bottom,
                  half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
                  EROSION);

  deviceImage = i_deviceImage + half_padding_bottom * xsize * ysize;

  // B_hat = closing - f
  subtraction(deviceImage, deviceOutput, xsize * ysize * zsize, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free device memory
  cudaFree(ii_deviceImage);
  cudaFree(i_deviceTmp);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
// Template instantiations for specific types
template void bottom_hat_on_device<int>(int*, int*, const int, const int, const int, const int,
                                        const int, const int, int*, int, int, int);
template void bottom_hat_on_device<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                                 const int, const int, const int, const int, int*,
                                                 int, int, int);
template void bottom_hat_on_device<float>(float*, float*, const int, const int, const int,
                                          const int, const int, const int, int*, int, int, int);

template <typename dtype>
void bottom_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                        int kernel_zsize) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;

  // Opening operation
  MorphChain closing = {DILATION, EROSION};
  morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                kernel_ysize, kernel_zsize, closing);

  // B_hat = closing - f
  subtraction_on_host(hostImage, hostOutput, size);
}
// Template instantiations for specific types
template void bottom_hat_on_host<int>(int*, int*, const int, const int, const int, int*, int, int,
                                      int);
template void bottom_hat_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                               const int, int*, int, int, int);
template void bottom_hat_on_host<float>(float*, float*, const int, const int, const int, int*, int,
                                        int, int);