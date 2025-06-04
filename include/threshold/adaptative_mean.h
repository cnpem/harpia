#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"

template <typename dtype>
void local_mean_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel);

template <typename in_dtype, typename out_dtype>
void adaptativeMeanThreshold3DGPU(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                     int nx, int ny, int nz, float weight);

template<typename in_dtype, typename out_dtype>
void adaptativeMeanThresholdChunked(in_dtype* hostImage, out_dtype* hostOutput, int xsize, int ysize, int zsize, float weight,int type3d, int flag_verbose,
                       float gpuMemory, int ngpus, int nx, int ny, int nz);