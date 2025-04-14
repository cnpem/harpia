#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "../../include/morphology/cuda_helper.h"

template <typename Func, typename dtype, typename... Args>
void chunkedExecutorGeodesic(Func func, int ncopies, const float safetyMargin, int ngpus, 
                             dtype* image, dtype* mask, dtype* output, const int xsize, 
                             const int ysize, const int zsize, const int verbose, Args... args) {

  dtype* i_ref = image;
  dtype* m_ref = mask;
  dtype* o_ref = output;

  // Get memory allocated by the func
  int sliceSize = xsize * ysize;
  size_t sliceBytes = static_cast<size_t>(sliceSize) * sizeof(dtype) * ncopies;

  // Get number of available devices
  int ngpus_available;
  CHECK(cudaGetDeviceCount(&ngpus_available));
  
  // Ajust number of gpus to available gpus
  if (ngpus_available < 1){
    if(verbose){
      printf("No gpus available. Cannot execute operations on the gpu.");
    }
    return;
  }
  else  if ((ngpus_available < ngpus) || (ngpus < 1)){
    if(verbose){
      printf("Number of gpus ajusted to maximun available gpus: %d.", ngpus_available);
    }
    ngpus = ngpus_available;
  }

  // Get free memory on the GPU with less memory in bytes
  size_t freeBytes, totalBytes, freeGpuBytes;
  CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  for (int i = 1; i < ngpus; i++) {
    cudaSetDevice(i);
    CHECK(cudaMemGetInfo(&freeGpuBytes, &totalBytes));
    if (freeGpuBytes < freeBytes) {
      freeBytes = freeGpuBytes;
    }
  }

  // How many slices fit in the GPU?
  if (safetyMargin > 1 || safetyMargin < 0) {
    fprintf(stderr,
            "Error: GPU %.2f memory occupancy is invalid. Choose a value between 0 and 1.\n",
            safetyMargin);
    return;
  }
  int chunkSize = static_cast<int>(freeBytes * safetyMargin / sliceBytes);
  int padding;
  if (zsize == 1) {
    padding = 0;
  } else {
    padding = 2;
  }
  int padding_top = 0;
  int padding_bottom = 0;

  // CASE: Not even one slice fits GPU mmemory
  if (chunkSize == 0) {
    fprintf(
        stderr,
        "Error: Not enough memory to fit even one slice. Adjust slice size or free up memory.\n");
    return;

    // CASE: intire input fits GPU memory (no padding)
  } else if (chunkSize >= zsize) {
    func(i_ref, m_ref, o_ref, xsize, ysize, zsize, verbose, padding_bottom, padding_top, args...);
    if (verbose) {
      printf("\nFinished processing!\n");
    }
    return;
  } else {
    chunkSize = (chunkSize - 2 * padding > 1) ? chunkSize - 2 * padding : 1;
  }
  if (verbose) {
    printf("MaxChunkSize:%d zsize:%d ngpus:%d\n", chunkSize, zsize, ngpus);
  }

  // CASE: break input into chunks (padding)

  // First chunk: padding only at the end
  int deviceCount = 0;
  int selectedDevice = deviceCount % ngpus;
  CHECK(cudaSetDevice(selectedDevice));
  cudaDeviceSynchronize();
  if (verbose) {
    printf("\niz:0 gpu:%d deviceCount:%d\n", selectedDevice, deviceCount);
  }
  padding_bottom = 0;
  padding_top = padding;

  func(i_ref, m_ref, o_ref, xsize, ysize, chunkSize, verbose, padding_bottom, padding_top, args...);
  i_ref += chunkSize * sliceSize;
  m_ref += chunkSize * sliceSize;
  o_ref += chunkSize * sliceSize;
  deviceCount += 1;

  // Middle chunks: padding at the beginning and at the end
  padding_bottom = padding;
  int iz = 0;
  for (iz = chunkSize; iz <= zsize - chunkSize; iz += chunkSize) {
    selectedDevice = deviceCount % ngpus;
    CHECK(cudaSetDevice(selectedDevice));
    cudaDeviceSynchronize();
    if (verbose) {
      printf("\niz:%d gpu:%d deviceCount:%d\n", iz, selectedDevice, deviceCount);
    }
    int remaining = zsize - iz - chunkSize;  // Check if this is the last chunk
    if (remaining <= 0) {
      // Possible last chunk: padding only at the begining
      padding_top = 0;
    }
    func(i_ref, m_ref, o_ref, xsize, ysize, chunkSize, verbose, padding_bottom, padding_top,
         args...);
    i_ref += chunkSize * sliceSize;  // Move to the next chunk
    m_ref += chunkSize * sliceSize;  // Move to the next chunk
    o_ref += chunkSize * sliceSize;
    deviceCount += 1;
  }

  // Last chunk: padding only at the begining
  int remaining = zsize - iz;
  selectedDevice = deviceCount % ngpus;
  CHECK(cudaSetDevice(selectedDevice));
  cudaDeviceSynchronize();
  if (verbose) {
    printf("\nremaining:%d gpu:%d deviceCount:%d\n", remaining, selectedDevice, deviceCount);
  }
  if (remaining > 0) {
    padding_top = 0;
    func(i_ref, m_ref, o_ref, xsize, ysize, remaining, verbose, padding_bottom, padding_top,
         args...);
  }

  if (verbose) {
    printf("\nFinished processing all chunks!\n");
  }
}
