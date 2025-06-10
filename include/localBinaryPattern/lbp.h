#ifndef LBP_CUH
#define LBP_CUH

#include <cuda.h>
#include <cuda_runtime.h>

// LBP kernel declaration
template <typename in_dtype>
__global__ void lbp(const in_dtype* devImage, float* devOutput, int xsize, int ysize, int zsize, int idz);

// Host wrapper
template <typename in_dtype>
void localBinaryPattern(in_dtype* hostImage, float* hostOutput, int xsize, int ysize, int zsize);

#endif // LBP_CUH
