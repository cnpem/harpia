#ifndef FRACTION_COUNTER_H
#define FRACTION_COUNTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

// Function prototypes
__global__ void fraction_counter(int* image, int* counter, int acumulator, int xsize, int ysize,
                                 int zsize);
__global__ void labels_fraction(int* image, int* counter, int acumulator, int xsize, int ysize,
                                int zsize);
void fraction(int* image, int* output, int xsize, int ysize, int zsize);

#endif  // FRACTION_COUNTER_H
