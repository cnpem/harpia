#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void Initialization2D(int* block_labels, int label_step, int xsize, int ysize);
__global__ void Merge2D(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize);
__global__ void CompressionLabels2D(int* block_labels, int label_step, int xsize, int ysize);
__global__ void FinalLabeling2D(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize);

__global__ void Initialization3D(int* block_labels, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize);
__global__ void Merge3D(int* image, int* block_labels, int image_step, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize);
__global__ void CompressionLabels3D(int* block_labels, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize);
__global__ void FinalLabeling3D(int* image, int* block_labels, int image_step, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize);

void connectedComponents(int* image, int* output, int xsize, int ysize, int zsize, bool type);

#endif // CONNECTED_COMPONENTS_H
