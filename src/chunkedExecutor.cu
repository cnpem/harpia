#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "../include/morphology/cuda_helper.h"

// Wrapper function
template <typename Func, typename dtype, typename... Args>
void chunkedExecutor(Func func, int ncopies, dtype* hostImage, dtype* hostOutput, const int xsize,
                     const int ysize, const int zsize, Args... args) {

  dtype* i_ref = hostImage;
  dtype* o_ref = hostOutput;

  const float safetyMargin = 0.2f;

  // Get memory allocated by the func
  int sliceSize = xsize * ysize;
  size_t sliceBytes = static_cast<size_t>(sliceSize) * sizeof(dtype) * ncopies;

  // Get free memory on the GPU in bytes
  size_t freeBytes;
  size_t totalBytes;
  CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  // How many slices fit in the GPU?
  int chunkSize = static_cast<int>(freeBytes * safetyMargin / sliceBytes);
  printf("MaxChunkSize:%d zsize:%d\n", chunkSize, zsize);

  if (chunkSize == 0) {
    fprintf(
        stderr,
        "Error: Not enough memory to fit even one slice. Adjust slice size or free up memory.\n");
    return;
  } else if (chunkSize > zsize) {
    chunkSize = zsize;
    printf("ActualChunkSize:%d zsize:%d\n", chunkSize, zsize);
  }

  int iz = 0;
  for (; iz <= zsize - chunkSize; iz += chunkSize) {
    printf("\niz:%d \n", iz);
    // Call the actual function with the rest of the arguments
    func(i_ref, o_ref, xsize, ysize, chunkSize, args...);
    i_ref += chunkSize * sliceSize;
    o_ref += chunkSize * sliceSize;
  }

  // Process the remaining slices, if any
  int remaining = zsize - iz;
  printf("\nremaining:%d \n", remaining);
  if (remaining > 0) {
    func(i_ref, o_ref, xsize, ysize, remaining, args...);
  }
}