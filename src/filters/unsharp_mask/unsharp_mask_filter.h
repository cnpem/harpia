
#ifndef UNSHARP_MASK_FILTER_H
#define UNSHARP_MASK_FILTER_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//CUDA kernel for the unsharp mask filter.
template<typename dtype>
void unsharp_mask_filtering(dtype* image, float* output, int rows, int cols, int depth, float sigma, float ammount, float threshold, bool type);

#endif // UNSHARP_MASK_FILTER_H