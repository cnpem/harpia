#ifndef CANNY_FILTER_H
#define CANNY_FILTER_H

#include <cuda_runtime.h>
#include <iostream>

#include "../common/convolution.h"
#include "gaussian_filter.h"

#define PI 3.141592653589793

// Function declarations

void get_horizontal_kernel_2d(float** kernel);

void get_vertical_kernel_2d(float** kernel);

__global__ void gradient_magnitude_direction_2d(float* image, float* magnitude, uint8_t* direction,
                                                float* horizontal_kernel, float* vertical_kernel,
                                                int xsize, int ysize, int idz);

__global__ void non_maximum_supression_2d(float* magnitude, uint8_t* direction, int xsize,
                                          int ysize, int idz);

__global__ void thresholding_2d(float* image, float low, float high, int xsize, int ysize, int idz);

__global__ void hysteresis_2d(float* image, int xsize, int ysize, int idz);

template <typename dtype>
void canny_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma,
                     float low_threshold, float high_threshold);

#endif  // CANNY_FILTER_H
