#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"

/**
 * @brief Performs a morphological chain operation on a binary image using the GPU.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param chain MorphChain structure containing the operations to be performed.
 * @param flag_verbose Flag for verbose output.
 */
template <typename dtype>
void morph_chain_binary_on_device(dtype* hostImage, dtype* hostOutput, int* kernel,
                                  int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                  const int xsize, const int ysize, const int zsize,
                                  MorphChain chain, const int flag_verbose) {

  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // set kernel dimension
  int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // malloc device global memory
  dtype *deviceImage, *deviceTmp, *deviceOutput;
  int* deviceKernel;
  CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // morphChain operation
  morph_binary(deviceImage, deviceTmp, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
               xsize, ysize, zsize, chain.operation1, flag_verbose);
  morph_binary(deviceTmp, deviceOutput, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
               xsize, ysize, zsize, chain.operation2, flag_verbose);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free device memory
  cudaFree(deviceTmp);
  cudaFree(deviceImage);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}

// Template instantiations for specific types
template void morph_chain_binary_on_device<int>(int*, int*, int*, int, int, int, const int,
                                                const int, const int, MorphChain, const int);
template void morph_chain_binary_on_device<unsigned int>(unsigned int*, unsigned int*, int*, int,
                                                         int, int, const int, const int, const int,
                                                         MorphChain, const int);
template void morph_chain_binary_on_device<uint16_t>(uint16_t*, uint16_t*, int*, int, int, int,
                                                     const int, const int, const int, MorphChain,
                                                     const int);

/**
 * @brief Performs a morphological chain operation on a binary image using the CPU.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param chain MorphChain structure containing the operations to be performed.
 */
template <typename dtype>
void morph_chain_binary_on_host(dtype* hostImage, dtype* hostOutput, int* kernel, int kernel_xsize,
                                int kernel_ysize, int kernel_zsize, const int xsize,
                                const int ysize, const int zsize, MorphChain chain) {

  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // allocate temporary memory
  dtype* hostTmp;
  hostTmp = (dtype*)malloc(nBytes);

  // set input data
  memset(hostTmp, 0, nBytes);

  // morphChain operation
  morph_binary_on_host(hostImage, hostTmp, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize,
                       ysize, zsize, chain.operation1);
  morph_binary_on_host(hostTmp, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize,
                       ysize, zsize, chain.operation2);

  // free temporary memory
  free(hostTmp);
}

// Template instantiations for specific types
template void morph_chain_binary_on_host<int>(int*, int*, int*, int, int, int, const int, const int,
                                              const int, MorphChain);
template void morph_chain_binary_on_host<unsigned int>(unsigned int*, unsigned int*, int*, int, int,
                                                       int, const int, const int, const int,
                                                       MorphChain);
template void morph_chain_binary_on_host<uint16_t>(uint16_t*, uint16_t*, int*, int, int, int,
                                                   const int, const int, const int, MorphChain);