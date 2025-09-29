#ifndef REMOVE_ISLANDS_H
#define REMOVE_ISLANDS_H

#include <cuda_runtime.h>

// Kernel function declarations
__global__ void label_counter_2d(int* image, int* counter, int xsize, int ysize);
__global__ void remove_2d(int* image, int* counter, int threshold, int xsize, int ysize);
__global__ void label_counter_3d(int* image, int* counter, int xsize, int ysize, int zsize);
__global__ void remove_3d(int* image, int* counter, int threshold, int xsize, int ysize, int zsize);

// Host function declaration
void remove_islands(int* image, int* output, int threshold, int xsize, int ysize, int zsize, bool type);

#endif // REMOVE_ISLANDS_H
