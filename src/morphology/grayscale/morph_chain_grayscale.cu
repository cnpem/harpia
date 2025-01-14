#include <stdio.h>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"

/**
 * @brief Performs a chain of grayscale morphological operations on the device.
 *
 * This function applies a series of morphological operations defined in a `MorphChain` on an image
 * using CUDA. It allocates memory on the device, transfers data, applies the operations, and then
 * copies the result back to the host.
 *
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param hostImage Pointer to the input grayscale image on the host.
 * @param hostOutput Pointer to the output image where the result will be stored on the host.
 * @param kernel Pointer to the kernel used for the morphological operations.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param chain A `MorphChain` structure containing the sequence of operations to be performed.
 * @param flag_verbose If non-zero, print verbose output about the grid and block dimensions.
 */
template <typename dtype>
void morph_chain_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                     const int ysize, const int zsize, const int flag_verbose,
                                     const int padding_bottom, const int padding_top, int* kernel,
                                     int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                     MorphChain chain) {
  // set input dimension
  size_t size = xsize * ysize * zsize;
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
  dtype *ii_hostImage, *ii_deviceImage, *i_deviceImage, *i_deviceTmp, *deviceTmp, *deviceOutput;
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
                  chain.operation1);

  deviceTmp = i_deviceTmp + half_padding_bottom * xsize * ysize;

  // Perform the second operation in the chain
  morph_grayscale(deviceTmp, deviceOutput, xsize, ysize, zsize, flag_verbose, half_padding_bottom,
                  half_padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize,
                  chain.operation2);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free device memory
  cudaFree(ii_deviceImage);
  cudaFree(i_deviceTmp);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
template void morph_chain_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                            const int, const int, const int,
                                                            const int, const int, int*, int, int,
                                                            int, MorphChain);
template void morph_chain_grayscale_on_device<int>(int*, int*, const int, const int, const int,
                                                   const int, const int, const int, int*, int, int,
                                                   int, MorphChain);
template void morph_chain_grayscale_on_device<float>(float*, float*, const int, const int,
                                                     const int, const int, const int, const int,
                                                     int*, int, int, int, MorphChain);

/**
 * @brief Performs a chain of grayscale morphological operations on the host.
 *
 * This function applies a sequence of morphological operations defined in a `MorphChain` on an
 * image using a CPU-based approach. It first allocates temporary memory, applies the operations
 * sequentially, and then frees the temporary memory.
 *
 * @tparam dtype The data type of the image pixels (e.g., unsigned int, int, float).
 * @param hostImage Pointer to the input grayscale image on the host.
 * @param hostOutput Pointer to the output image where the result will be stored on the host.
 * @param kernel Pointer to the kernel used for the morphological operations.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param chain A `MorphChain` structure containing the sequence of operations to be performed.
 */
template <typename dtype>
void morph_chain_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                   const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                   int kernel_ysize, int kernel_zsize, MorphChain chain) {

  // set input dimension
  size_t size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // allocate temporary memory
  dtype* hostTmp;
  hostTmp = (dtype*)malloc(nBytes);

  // initialize temporary memory
  memset(hostTmp, 0, nBytes);

  // Perform the first operation in the chain
  morph_grayscale_on_host(hostImage, hostTmp, xsize, ysize, zsize, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, chain.operation1);

  // Perform the second operation in the chain
  morph_grayscale_on_host(hostTmp, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, chain.operation2);

  // Free temporary memory
  free(hostTmp);
}
template void morph_chain_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                          const int, const int, int*, int, int, int,
                                                          MorphChain);
template void morph_chain_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*,
                                                 int, int, int, MorphChain);
template void morph_chain_grayscale_on_host<float>(float*, float*, const int, const int, const int,
                                                   int*, int, int, int, MorphChain);