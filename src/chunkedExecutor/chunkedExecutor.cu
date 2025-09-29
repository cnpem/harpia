#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "../../include/morphology/cuda_helper.h"

template <typename Func, typename in_dtype, typename out_dtype, typename... Args>
void chunkedExecutor(Func func, int ncopies, const float safetyMargin, int ngpus, 
                     in_dtype* image, out_dtype* output, const int xsize, const int ysize, const int zsize, 
                     const int verbose, Args... args) {

  in_dtype* i_ref = image;
  out_dtype* o_ref = output;

  // Calculate slice size and memory usage
  int sliceSize = xsize * ysize;
  size_t sliceBytes = static_cast<size_t>(sliceSize) * sizeof(in_dtype) * ncopies;

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

  // Determine the GPU with the least available memory
  size_t freeBytes, totalBytes, freeGpuBytes;
  cudaSetDevice(0);
  CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  for (int i = 1; i < ngpus; i++) {
    cudaSetDevice(i);
    CHECK(cudaMemGetInfo(&freeGpuBytes, &totalBytes));
    if (freeGpuBytes < freeBytes) {
      freeBytes = freeGpuBytes;
    }
  }

  // Determine chunk size based on available memory
  int chunkSize = static_cast<int>(freeBytes * safetyMargin / sliceBytes);
  if (verbose) {
    printf("MaxChunkSize:%d zsize:%d ngpus:%d\n", chunkSize, zsize, ngpus);
  }

  if (chunkSize == 0) {
    fprintf(
        stderr,
        "Error: Not enough memory to fit even one slice. Adjust slice size or free up memory.\n");
    return;
  } else if (chunkSize > zsize) {
    chunkSize = zsize;
    if (verbose) {
      printf("ActualChunkSize:%d\n", chunkSize);
    }
  }

  // Process image in chunks
  int iz = 0;
  int deviceCount = 0;
  int selectedDevice;
  for (; iz <= zsize - chunkSize; iz += chunkSize) {
    selectedDevice = deviceCount % ngpus;
    CHECK(cudaSetDevice(selectedDevice));
    cudaDeviceSynchronize();
    if (verbose) {
      printf("Processing chunk: iz=%d, chunkSize=%d, device=%d\n", iz, chunkSize, selectedDevice);
    }
    if (verbose) {
      printf("i_ref ptr: %p | max ptr: %p\n", (void*)i_ref, (void*)(image + zsize * sliceSize));
      printf("o_ref ptr: %p | max ptr: %p\n", (void*)o_ref, (void*)(output + zsize * sliceSize));
    }
    func(i_ref, o_ref, xsize, ysize, chunkSize, verbose, args...);
    i_ref += chunkSize * sliceSize;
    o_ref += chunkSize * sliceSize;
    deviceCount += 1;
  }

  // Process remaining slices
  int remaining = zsize - iz;
  selectedDevice = deviceCount % ngpus;
  CHECK(cudaSetDevice(selectedDevice));
  cudaDeviceSynchronize();
  if (verbose) {
    printf("\nremaining:%d gpu:%d deviceCount:%d\n", remaining, selectedDevice, deviceCount);
  }
  if (remaining > 0) {
    func(i_ref, o_ref, xsize, ysize, remaining, verbose, args...);
  }
  if (verbose) {
    printf("\nFinished processing all chunks!\n");
  }
}