#ifndef CCL_H_
#define CCL_H_

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ bool HasBit(int bitmask, int bit);

__global__ void Initialization(int* block_labels, int label_step, int xsize, int ysize);

__global__ void Merge(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize);

__global__ void CompressionLabels(int* block_labels, int label_step, int xsize, int ysize);

__global__ void FinalLabeling(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize);

void connectedComponents(int* image, int* output, int xsize, int ysize);

#endif /* CCL_H_ */