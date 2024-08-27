#include <cuda_runtime.h>
#include <iostream>
#include "../common/kernels.h"

template <typename dtype>
void local_mean_threshold(dtype* image, float* output, float weight, int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel);