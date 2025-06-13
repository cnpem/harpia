#ifndef GRADIENT2D_H
#define GRADIENT2D_H

#include <cuda_runtime.h>

// CUDA kernel: Computes finite difference gradient in 2D slices
template<typename in_dtype>
__global__ void gradient2D(in_dtype* devImage, float* devOutput,
                           int xsize, int ysize, int zsize,
                           int idz, int axis, float step);

// Host function: Calls the CUDA kernel slice-by-slice
template <typename in_dtype>
void gradient(in_dtype* hostImage, float* hostOutput,
              int xsize, int ysize, int zsize,
              int axis, float step);

#endif  // GRADIENT2D_H

