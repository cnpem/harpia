#include <cuda_runtime.h>
#include <stdio.h>
#include "../../include/morphology/cuda_helper.h"

void throw_on_cuda_error(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    printf("Error: %s: %d", __FILE__, __LINE__);
    printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));
    exit(1);
  }
}

void test_check_device_info() {
  printf("\nCheck device info:\n\n");

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev = 0, driverVersion = 0, runtimeVersion = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp deviceProp;

  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Device %d: \"%s\"\n", dev, deviceProp.name);

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major,
         deviceProp.minor);
  printf(
      "  Total amount of global memory:                 %.2f MBytes (%llu "
      "bytes)\n",
      (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
      (unsigned long long)deviceProp.totalGlobalMem);
  printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
  printf("  Memory Clock rate:                             %.0f MHz\n",
         deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
  }

  printf(
      "  Max Texture Dimension Size (x,y,z)             1D=(%d), "
      "2D=(%d,%d), 3D=(%d,%d,%d)\n",
      deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
      deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf(
      "  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
      "2D=(%d,%d) x %d\n",
      deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
      deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
      deviceProp.maxTexture2DLayered[2]);
  printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %lu bytes\n",
         deviceProp.sharedMemPerBlock);
  printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n", deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n",
         deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
  printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
         deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
         deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
}

void checkGpuMem(size_t allocatedBytes) {
  float free_g, total_g, used_g, allocated_g;
  size_t free_t, total_t;

  // Get free and total memory on the GPU
  cudaMemGetInfo(&free_t, &total_t);

  // Convert bytes to gigabytes (1 GB = 1024^3 bytes)
  free_g = static_cast<float>(free_t) / 1073741824.0f;
  total_g = static_cast<float>(total_t) / 1073741824.0f;
  used_g = total_g - free_g;
  allocated_g = static_cast<float>(allocatedBytes) / 1073741824.0f;
  // Print memory usage in GB
  printf(
      // "mem free: %f GB (%zu bytes) | mem total: %f GB (%zu bytes) | mem used: %f GB | mem "
      // "allocated: %f GB\n",
      "mem free:%fGB  total:%fGB  used:%fGB  allocated:%fGB\n", free_g, total_g, used_g,
      allocated_g);
}