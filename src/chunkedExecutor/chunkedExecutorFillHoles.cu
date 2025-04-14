#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "../../include/morphology/cuda_helper.h"
#include "../../include/morphology/logical_or_binary.h"

// Wrapper function
// ToDo: The fill holes operation has no fixed number of iterations, since it uses a convergency 
// operation of reconstruction. Because of that, it is not possible to determine a padding value to
// break the operation in chunks, the needed padding would have t be infine.
// But the operation can be executed wihtout padding and overlapping chunks. This way there will be 
// border issues with possibly non-filled holes, but the overlapped chunk will have the border as
// interior area, and this hole will be filled. A silmpe or operation can unite this results.
// This solution is implemented on annotat3d backend in pyhton, but it could be done in cuda, it
// isn't working yet here.

template <typename Func, typename dtype, typename... Args>
void chunkedExecutorFillHoles(Func func, int ncopies, const float safetyMargin, int ngpus, 
                              dtype* image, dtype* output, int padding, const int xsize, 
                              const int ysize, const int zsize, const int verbose, Args... args) {

  // set output initial data
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(dtype);

  dtype* margins = (dtype*)malloc(nBytes);

  dtype* i_ref = image;
  dtype* o_ref = output;
  dtype* m_ref = margins;

  memset(o_ref, 0, nBytes);
  memset(m_ref, 0, nBytes);

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
  int chunkSize = static_cast<int>(freeBytes * safetyMargin / sliceBytes);
  if (verbose) {
    printf("MaxChunkSize:%d zsize:%d ngpus:%d\n", chunkSize, zsize, ngpus);
  }

  if (chunkSize == 0) {
    fprintf(
        stderr,
        "Error: Not enough memory to fit even one slice. Adjust slice size or free up memory.\n");
    return;
  } else if (chunkSize >= zsize) {
    func(i_ref, o_ref, xsize, ysize, zsize, 0, args...);
    if (verbose) {
      printf("\nFinished processing!\n");
    }
    return;
  }

  // Execute output
  int iz = 0;
  int deviceCount = 0;
  int selectedDevice;
  for (; iz <= zsize - chunkSize; iz += chunkSize) {
    selectedDevice = deviceCount % ngpus;
    CHECK(cudaSetDevice(selectedDevice));
    if (verbose) {
      printf("\niz:%d gpu:%d deviceCount:%d\n", iz, selectedDevice, deviceCount);
    }
    func(i_ref, o_ref, xsize, ysize, chunkSize, 0, args...);
    i_ref += chunkSize * sliceSize;
    o_ref += chunkSize * sliceSize;
    deviceCount += 1;
  }

  // Process the remaining slices, if any
  int remaining = zsize - iz;
  selectedDevice = deviceCount % ngpus;
  CHECK(cudaSetDevice(selectedDevice));
  if (verbose) {
    printf("\nremaining:%d gpu:%d deviceCount:%d\n", remaining, selectedDevice, deviceCount);
  }
  if (remaining > 0) {
    func(i_ref, o_ref, xsize, ysize, remaining, 0, args...);
    deviceCount += 1;
  }
  if (verbose) {
    printf("\nFinished processing all chunks!\n");
  }

  // Execute margins

  padding = (padding < chunkSize/2) ? padding : chunkSize/2;
  iz = chunkSize-padding;
  //reset pointers
  i_ref = image;
  i_ref += iz * sliceSize;
  m_ref += iz * sliceSize;
  for (; iz <= zsize - chunkSize - padding; iz += chunkSize) {
    selectedDevice = deviceCount % ngpus;
    CHECK(cudaSetDevice(selectedDevice));
    if (verbose) {
      printf("\niz:%d gpu:%d deviceCount:%d\n", iz, selectedDevice, deviceCount);
    }
    func(i_ref, m_ref, xsize, ysize, 2*padding, 0, args...);
    i_ref += chunkSize * sliceSize;
    m_ref += chunkSize * sliceSize;
    deviceCount += 1;
  }

  // Process the remaining margins, if any
  // Use previous calculated remaining value
  selectedDevice = deviceCount % ngpus;
  CHECK(cudaSetDevice(selectedDevice));
  if (verbose) {
    printf("\nremaining:%d gpu:%d deviceCount:%d\n", remaining+padding, selectedDevice, deviceCount);
  }
  if (remaining > 0) {
    func(i_ref, m_ref, xsize, ysize, remaining+padding, 0, args...);
  }
  if (verbose) {
    printf("\nFinished processing all margins!\n");
  }

  // Synchronize all devices first
  for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i); // Set the current device
        cudaError_t err = cudaDeviceSynchronize(); // Synchronize the device
        if (err != cudaSuccess) {
            std::cerr << "Error synchronizing device " << i << ": " 
                      << cudaGetErrorString(err) << std::endl;
        }
  }

  // Unite output and margins
  chunkedExecutor(logical_or_on_device<dtype>, 2, safetyMargin, ngpus, margins, output,
                  xsize, ysize, zsize, 0);

}