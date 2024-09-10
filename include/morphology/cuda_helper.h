#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>

void throw_on_cuda_error(cudaError_t error, const char* file, int line);

void test_check_device_info();

void checkGpuMem(size_t allocatedBytes);

#define CHECK(call) \
  { throw_on_cuda_error((call), __FILE__, __LINE__); };

#endif  //CUDA_HELPER_H