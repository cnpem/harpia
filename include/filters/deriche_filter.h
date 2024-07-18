#ifndef DERICHE_FILTER_H
#define DERICHE_FILTER_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include"../common/convolution.h"
#define PI 3.141592653589793

void get_deriche_vertical_kernel_2d(float** kernel, float alpha, int rows, int cols);
void get_deriche_horizontal_kernel_2d(float** kernel, float alpha, int rows, int cols);
void get_deriche_vertical_kernel_3d(float** kernel, float alpha, float beta, int rows, int cols, int depth);

__global__ void deriche_gradient_magnitude_direction_2d(float* image, float* magnitude, uint8_t* direction,
                                                        float* horizontal_kernel, float* vertical_kernel,
                                                        int rows, int cols, int idz);

__global__ void deriche_non_maximum_supression_2d(float* magnitude, uint8_t* direction, int rows, int cols, int idz);
__global__ void deriche_thresholding_2d(float* image, float low, float high, int rows, int cols, int idz);
__global__ void deriche_hysteresis_2d(float* image, int rows, int cols, int idz);

template<typename dtype>
void deriche_filtering(dtype* image, float* output,
                       int rows, int cols , int depth,
                       int rows_kernel, int cols_kernel, 
                       float alpha, float low_threshold, float high_threshold);

#endif // DERICHE_FILTER_H


