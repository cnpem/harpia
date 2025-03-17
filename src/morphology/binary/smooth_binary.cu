#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/smooth_binary.h"

template <typename dtype>
void smooth_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, const int flag_verbose, const int padding_bottom,
                             const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize) {

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);
  size_t nBytes_padding = xsize * ysize * (padding_bottom + padding_top) * sizeof(dtype);

  size_t nBytes_input = nBytes + nBytes_padding;

  int quarter_padding_bottom = padding_bottom / 4;
  int quarter_padding_top = padding_top / 4;

  // set kernel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // malloc device global memory
  dtype *i_hostImage, *deviceImage, *deviceTmp;
  int* deviceKernel;

  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes_input));
  CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes_input));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  //Save original pointer for memmory deallocation
  dtype *original_deviceImage = deviceImage;
  dtype *original_deviceTmp = deviceTmp;

  // transfer input + padding
  i_hostImage = hostImage - padding_bottom * xsize * ysize;

  CHECK(cudaMemcpy(deviceImage, i_hostImage, nBytes_input, cudaMemcpyHostToDevice));

  // advance pointers 1
  deviceImage = deviceImage + quarter_padding_bottom * xsize * ysize;
  deviceTmp = deviceTmp + quarter_padding_bottom * xsize * ysize;

  // Perform the first operation in the chain
  morph_binary(deviceImage, deviceTmp, xsize, ysize,
               zsize + 3 * quarter_padding_top + 3 * quarter_padding_bottom, flag_verbose,
               quarter_padding_bottom, quarter_padding_top, deviceKernel, kernel_xsize,
               kernel_ysize, kernel_zsize, EROSION);

  // advance pointers 2
  deviceImage = deviceImage + quarter_padding_bottom * xsize * ysize;
  deviceTmp = deviceTmp + quarter_padding_bottom * xsize * ysize;

  // Perform the second operation in the chain
  morph_binary(deviceTmp, deviceImage, xsize, ysize,
               zsize + 2 * quarter_padding_top + 2 * quarter_padding_bottom, flag_verbose,
               quarter_padding_bottom, quarter_padding_top, deviceKernel, kernel_xsize,
               kernel_ysize, kernel_zsize, DILATION);

  // advance pointers 3
  deviceImage = deviceImage + quarter_padding_bottom * xsize * ysize;
  deviceTmp = deviceTmp + quarter_padding_bottom * xsize * ysize;

  // Perform the first operation in the chain
  morph_binary(deviceImage, deviceTmp, xsize, ysize,
               zsize + quarter_padding_top + quarter_padding_bottom, flag_verbose,
               quarter_padding_bottom, quarter_padding_top, deviceKernel, kernel_xsize,
               kernel_ysize, kernel_zsize, DILATION);

  // advance pointers 4
  deviceImage = deviceImage + quarter_padding_bottom * xsize * ysize;
  deviceTmp = deviceTmp + quarter_padding_bottom * xsize * ysize;

  // Perform the second operation in the chain
  morph_binary(deviceTmp, deviceImage, xsize, ysize, zsize, flag_verbose, quarter_padding_bottom,
               quarter_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
               EROSION);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceImage, nBytes, cudaMemcpyDeviceToHost));

  // free device memory
  cudaFree(original_deviceImage);
  cudaFree(original_deviceTmp);
  cudaFree(deviceKernel);
}

// Template instantiations for specific types
template void smooth_binary_on_device<int>(int*, int*, const int, const int, const int, const int,
                                           const int, const int, int*, int, int, int);
template void smooth_binary_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                    const int, const int, const int, const int,
                                                    const int, int*, int, int, int);
template void smooth_binary_on_device<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                               const int, const int, const int, int*, int, int,
                                               int);
template void smooth_binary_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                const int, const int, const int, const int, int*,
                                                int, int, int);
                                               
template <typename dtype>
void smooth_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                           int kernel_zsize) {

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // allocate temporary memory
  dtype* hostTmp;
  hostTmp = (dtype*)malloc(nBytes);

  //Opening
  morph_binary_on_host(hostImage, hostTmp, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                       kernel_zsize, EROSION);
  morph_binary_on_host(hostTmp, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                       kernel_zsize, DILATION);
  //Closing
  morph_binary_on_host(hostOutput, hostTmp, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                       kernel_zsize, DILATION);
  morph_binary_on_host(hostTmp, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                       kernel_zsize, EROSION);

  // free temporary memory
  free(hostTmp);
}

// Template instantiations for specific types
template void smooth_binary_on_host<int>(int*, int*, const int, const int, const int, int*, int,
                                         int, int);
template void smooth_binary_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                  const int, const int, int*, int, int, int);
template void smooth_binary_on_host<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                             int*, int, int, int);
template void smooth_binary_on_host<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                              int*, int, int, int);