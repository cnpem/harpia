#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include<iostream>
#include<cuda_runtime.h>

template<typename dtype>
__device__ void convolution_2d(dtype* input,
                                float* output,
                                float* kernel,
                                int i, int j,
                                int rows, int cols,
                                int rows_kernel, int cols_kernel);

template<typename dtype>
__device__ void convolution_3d(dtype* input,
                               float* output,
                               float* kernel,
                               int i, int j, int k,
                               int rows, int cols, int depth,
                               int rows_kernel, int cols_kernel, int depth_kernel);

#endif // CONVOLUTION_H
