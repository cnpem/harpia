#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"

template <typename dtype>
void morph_chain_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                  const int ysize, const int zsize, const int flag_verbose,
                                  const int padding_bottom, const int padding_top, int* kernel,
                                  int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                  MorphChain chain) {

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

  // Set kernel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // Malloc device global memory
  dtype *ii_hostImage, *ii_deviceImage, *i_deviceImage, *i_deviceTmp, *deviceTmp, *deviceOutput;
  int* deviceKernel;

  CHECK(cudaMalloc((dtype**)&ii_deviceImage, nBytes_input));
  CHECK(cudaMalloc((dtype**)&i_deviceTmp, nBytes_tmp));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // Adjust input pointer to account for padding
  ii_hostImage = hostImage - padding_bottom * xsize * ysize;

  // Transfer input plus padding
  CHECK(cudaMemcpy(ii_deviceImage, ii_hostImage, nBytes_input, cudaMemcpyHostToDevice));

  // Adjust device pointer to exclude padding
  i_deviceImage = ii_deviceImage + half_padding_bottom * xsize * ysize;

  // Perform the first operation in the morphological chain
  morph_binary(i_deviceImage, i_deviceTmp, xsize, ysize,
               zsize + half_padding_top + half_padding_bottom, flag_verbose, half_padding_bottom,
               half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
               chain.operation1);

  deviceTmp = i_deviceTmp + half_padding_bottom * xsize * ysize;

  // Perform the second operation in the morphological chain
  morph_binary(deviceTmp, deviceOutput, xsize, ysize, zsize, flag_verbose, half_padding_bottom,
               half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
               chain.operation2);
 
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));
 
  // Free device memory
  cudaFree(ii_deviceImage);
  cudaFree(i_deviceTmp);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
template void morph_chain_binary_on_device<int>(int*, int*, const int, const int, const int,
                                                const int, const int, const int, int*, int, int,
                                                int, MorphChain);
template void morph_chain_binary_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                         const int, const int, const int, const int,
                                                         const int, int*, int, int, int,
                                                         MorphChain);
template void morph_chain_binary_on_device<int16_t>(int16_t*, int16_t*, const int, const int,
                                                    const int, const int, const int, const int,
                                                    int*, int, int, int, MorphChain);
template void morph_chain_binary_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                     const int, const int, const int, const int,
                                                     int*, int, int, int, MorphChain);

template <typename dtype>
void morph_chain_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                int kernel_ysize, int kernel_zsize, MorphChain chain) {

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Allocate temporary memory
  dtype* hostTmp;
  hostTmp = (dtype*)malloc(nBytes);

  // Perform the first operation in the chain
  morph_binary_on_host(hostImage, hostTmp, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                       kernel_zsize, chain.operation1);
  
  // Perform the second operation in the chain
  morph_binary_on_host(hostTmp, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
  kernel_zsize, chain.operation2);

  // Free temporary memory
  free(hostTmp);
}
template void morph_chain_binary_on_host<int>(int*, int*, const int, const int, const int, int*,
                                              int, int, int, MorphChain);
template void morph_chain_binary_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                       const int, const int, int*, int, int, int,
                                                       MorphChain);
template void morph_chain_binary_on_host<int16_t>(int16_t*, int16_t*, const int, const int,
                                                  const int, int*, int, int, int, MorphChain);
template void morph_chain_binary_on_host<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                   const int, int*, int, int, int, MorphChain);