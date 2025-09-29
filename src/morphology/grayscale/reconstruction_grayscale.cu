#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>  // For float, unsigned int
#include <iostream>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/compare_arrays_grayscale.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/geodesic_morph_grayscale.h"
#include "../../../include/morphology/reconstruction_grayscale.h"

template <typename dtype>
void reconstruction_grayscale(dtype* deviceMarker, dtype* deviceMask, dtype* deviceOutput,
                              const int xsize, const int ysize, const int zsize, MorphOp operation,
                              const int flag_verbose) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Reconstruction: iterate geodesic erosion/dilation until convergency
  int hostFlagConverged = 1;
  int* deviceFlagConverged;
  CHECK(cudaMalloc((int**)&deviceFlagConverged, sizeof(int)));
  CHECK(cudaMemcpy(deviceFlagConverged, &hostFlagConverged, sizeof(int), cudaMemcpyHostToDevice));

  do {
    // Reconstruction step
    geodesic_morph_grayscale(deviceMarker, deviceMask, deviceOutput, xsize, ysize, zsize,
                             flag_verbose, 0, 0, operation);

    // Check convergency
    cudaMemset(deviceFlagConverged, 1,
               sizeof(int));  //compare_arrays_grayscale() initial output value MUST be 1 (true)
    compare_arrays_grayscale(deviceMarker, deviceOutput, deviceFlagConverged, size, flag_verbose);

    // Copy data to the next iteration
    CHECK(cudaMemcpy(deviceMarker, deviceOutput, nBytes, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(&hostFlagConverged, deviceFlagConverged, sizeof(int), cudaMemcpyDeviceToHost));
  } while (!hostFlagConverged);
}
template void reconstruction_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                            MorphOp, const int);
template void reconstruction_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                     const int, const int, const int, MorphOp,
                                                     const int);
template void reconstruction_grayscale<float>(float*, float*, float*, const int, const int,
                                              const int, MorphOp, const int);


template <typename dtype>
void reconstruction_grayscale_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                        const int xsize, const int ysize, const int zsize,
                                        MorphOp operation, const int flag_verbose) {
  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
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

  reconstruction_grayscale(deviceMarker, deviceMask, deviceOutput, xsize, ysize, zsize, operation,
                           flag_verbose);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  cudaFree(deviceMarker);
  cudaFree(deviceOutput);
}
template void reconstruction_grayscale_on_device<int>(int*, int*, int*, const int, const int,
                                                      const int, MorphOp, const int);
template void reconstruction_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*,
                                                               unsigned int*, const int, const int,
                                                               const int, MorphOp, const int);
template void reconstruction_grayscale_on_device<float>(float*, float*, float*, const int,
                                                        const int, const int, MorphOp, const int);


template <typename dtype>
void reconstruction_grayscale_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                      const int xsize, const int ysize, const int zsize,
                                      MorphOp operation) {
  int flagConverged = 0;

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // allocate marker memory
  dtype* marker;
  marker = (dtype*)malloc(nBytes);
  memcpy(marker, hostImage, nBytes);

  do {
    geodesic_morph_grayscale_on_host(marker, hostMask, hostOutput, xsize, ysize, zsize, operation);

    compare_arrays_grayscale_on_host(marker, hostOutput, &flagConverged, size);
    memcpy(marker, hostOutput, nBytes);

  } while (!flagConverged);

  // free host memorys
  free(marker);
}
template void reconstruction_grayscale_on_host<int>(int*, int*, int*, const int, const int,
                                                    const int, MorphOp);
template void reconstruction_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*,
                                                             unsigned int*, const int, const int,
                                                             const int, MorphOp);
template void reconstruction_grayscale_on_host<float>(float*, float*, float*, const int, const int,
                                                      const int, MorphOp);
