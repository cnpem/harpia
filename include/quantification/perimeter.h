#ifndef PERIMETER_COUNTER_H
#define PERIMETER_COUNTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

// Function prototypes
__device__ void isPerimeter(int* image, u_int* counter, int idx, int idy, int idz, int xsize,
                            int ysize, int zsize);
__global__ void perimeter_counter(int* image, u_int* counter, int xsize, int ysize, int zsize);
void perimeter(int* image, u_int* output, int xsize, int ysize, int zsize);

#endif  // PERIMETER_COUNTER_H
