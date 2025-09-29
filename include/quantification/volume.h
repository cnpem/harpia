#ifndef VOLUME_COUNTER_H
#define VOLUME_COUNTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

// Function prototypes
__device__ void isVolume(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize,
                         int ysize, int zsize);
__global__ void volume_counter(int* image, unsigned int* counter, int idz, int xsize, int ysize,
                               int zsize);
void volume(int* image, unsigned int* output, int xsize, int ysize, int zsize);

#endif  // VOLUME_COUNTER_H
