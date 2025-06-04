#ifndef NIBLACK_H
#define NIBLACK_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"
#include "../filters/mean_filter.h"
// 2D Niblack Kernel
template <typename dtype>
__global__ void niblack_kernel_2d(dtype* image, float* output, float weight, int rows, int cols,
                                  int idz, int rows_kernel, int cols_kernel);

// 3D Niblack Kernel
template <typename dtype>
__global__ void niblack_kernel_3d(dtype* image, float* output, float weight, int rows, int cols,
                                  int depth, int rows_kernel, int cols_kernel, int depth_kernel);

// Niblack Threshold Function
template <typename dtype>
void niblack_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
                       int rows_kernel, int cols_kernel, int depth_kernel);


template <typename in_dtype, typename out_dtype>
void niblackThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                     int nx, int ny, int nz, float weight);

template<typename in_dtype, typename out_dtype>
void niblackThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,float weight, int type3d, int flag_verbose,
                       float gpuMemory, int ngpus, int nx, int ny, int nz);

#endif  //NIBLACK_H