#ifndef AREA_SURFACE_COUNTER_H
#define AREA_SURFACE_COUNTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

// Function prototypes
__device__ void isArea(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize,
                       int ysize, int zsize);
__device__ void isSurface(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize,
                          int ysize, int zsize);
__global__ void area_counter(int* image, unsigned int* counter, int idz, int xsize, int ysize,
                             int zsize);
__global__ void surface_area_counter(int* image, unsigned int* counter, int idz, int xsize,
                                     int ysize, int zsize);
void area(int* image, unsigned int* output, int xsize, int ysize, int zsize, bool type);

#endif  // AREA_SURFACE_COUNTER_H
