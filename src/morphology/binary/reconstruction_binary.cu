#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include <iostream>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/compare_arrays_binary.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/geodesic_morph_binary.h"
#include "../../../include/morphology/reconstruction_binary.h"

template <typename dtype>
void reconstruction_binary(dtype* deviceMarker, dtype* deviceOutput, const int xsize,
                           const int ysize, const int zsize, dtype* deviceMask, MorphOp operation,
                           const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Reconstruction: iterate geodesic erosion/dilation until convergency
  int hostFlagConverged = 1;
  int* deviceFlagConverged;
  CHECK(cudaMalloc((int**)&deviceFlagConverged, sizeof(int)));
  CHECK(cudaMemcpy(deviceFlagConverged, &hostFlagConverged, sizeof(int), cudaMemcpyHostToDevice));

  do {
    //Reconstruction step
    geodesic_morph_binary(deviceMarker, deviceOutput, xsize, ysize, zsize, deviceMask, operation,
                          flag_verbose);

    //Check convergency
    cudaMemset(deviceFlagConverged, 1,
               sizeof(int));  //compare_arrays_binary() initial output value MUST be 1 (true)
    compare_arrays_binary(deviceMarker, deviceOutput, deviceFlagConverged, size, flag_verbose);

    //Copy data to the next iteration
    CHECK(cudaMemcpy(deviceMarker, deviceOutput, nBytes, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(&hostFlagConverged, deviceFlagConverged, sizeof(int), cudaMemcpyDeviceToHost));
  } while (!hostFlagConverged);
}
template void reconstruction_binary<int>(int*, int*, const int, const int, const int, int*, MorphOp,
                                         const int);
template void reconstruction_binary<unsigned int>(unsigned int*, unsigned int*, const int,
                                                  const int, const int, unsigned int*, MorphOp,
                                                  const int);
template void reconstruction_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                              uint16_t*, MorphOp, const int);

/**
 * @brief Perform recosntruction by erosion/dilation operation on the entire image using the GPU. 
 * This function is meant to be called from host and slide the geodesic_morph_binary kernel function 
 * through all pixels.
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
void reconstruction_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                     const int ysize, const int zsize, dtype* hostMask,

                                     MorphOp operation, const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // malloc device global memory
  dtype *deviceMarker, *deviceOutput, *deviceMask;
  CHECK(cudaMalloc((dtype**)&deviceMarker, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceMask, nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceMarker, hostImage, nBytes,
                   cudaMemcpyHostToDevice));  //the initial marker is the input image
  CHECK(cudaMemcpy(deviceMask, hostMask, nBytes, cudaMemcpyHostToDevice));

  reconstruction_binary(deviceMarker, deviceOutput, xsize, ysize, zsize, deviceMask, operation,
                        flag_verbose);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free device memorys
  cudaFree(deviceMarker);
  cudaFree(deviceOutput);
}
template void reconstruction_binary_on_device<int>(int*, int*, const int, const int, const int,
                                                   int*, MorphOp, const int);
template void reconstruction_binary_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                            const int, const int, unsigned int*,
                                                            MorphOp, const int);
template void reconstruction_binary_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                        const int, uint16_t*, MorphOp, const int);

/**
 * @brief Perform recosntruction by erosion/dilation operation on the entire image using the CPU. 
 * This function is used to check GPU results correctness.
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
void reconstruction_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                   const int ysize, const int zsize, dtype* hostMask,
                                   MorphOp operation) {

  int flagConverged = 0;

  // set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // allocate marker memory
  dtype* marker;
  marker = (dtype*)malloc(nBytes);
  memcpy(marker, hostImage, nBytes);

  do {
    geodesic_morph_binary_on_host(marker, hostOutput, xsize, ysize, zsize, hostMask, operation);

    compare_arrays_binary_on_host(marker, hostOutput, &flagConverged, size);
    memcpy(marker, hostOutput, nBytes);

  } while (!flagConverged);

  // free host memorys
  free(marker);
}
template void reconstruction_binary_on_host<int>(int*, int*, const int, const int, const int, int*,
                                                 MorphOp);
template void reconstruction_binary_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                          const int, const int, unsigned int*,
                                                          MorphOp);
template void reconstruction_binary_on_host<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                      const int, uint16_t*, MorphOp);
