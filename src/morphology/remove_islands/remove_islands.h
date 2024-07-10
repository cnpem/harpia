#ifndef REMOVE_ISLANDS_H
#define REMOVE_ISLANDS_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Function declarations
__global__ void label_counter(int* image, int* counter, int xsize, int ysize);
__global__ void remove(int* image, int* counter, int threshold, int xsize, int ysize);
void remove_islands(int* image, int* output, int threshold, int xsize, int ysize);

#endif // REMOVE_ISLANDS_H
