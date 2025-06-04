
#ifndef SAUVOLA_H
#define SAUVOLA_H

#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"

// 2D Sauvola Kernel
template <typename dtype>
__global__ void sauvola_kernel_2d(dtype* image, float* output, float weight, dtype range, int rows,
                                  int cols, int idz, int rows_kernel, int cols_kernel);

// 3D Sauvola Kernel
template <typename dtype>
__global__ void sauvola_kernel_3d(dtype* image, float* output, float weight, dtype range, int rows,
                                  int cols, int depth, int rows_kernel, int cols_kernel,
                                  int depth_kernel);

// Sauvola Threshold Function
template <typename dtype>
void sauvola_threshold(dtype* image, float* output, float weight, dtype range, int rows, int cols,
                       int depth, int rows_kernel, int cols_kernel, int depth_kernel);

template <typename in_dtype, typename out_dtype>
void sauvolaThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                     int nx, int ny, int nz, float weight, in_dtype range);

template<typename in_dtype, typename out_dtype>
void sauvolaThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize,float weight, in_dtype range,int type3d, int flag_verbose,
                       float gpuMemory, int ngpus, int nx, int ny, int nz);

#endif  // SAUVOLA_H