#ifndef UNSHARP_MASK_FILTER_H
#define UNSHARP_MASK_FILTER_H

#include <cuda_runtime.h>
#include <iostream>
#include "gaussian_filter.h"
// Function template for unsharp mask filtering
template <typename dtype>
void unsharp_mask_filtering(dtype* image, float* output, int xsize, int ysize, int zsize,
                            float sigma, float ammount, float threshold, bool type);

#endif  // UNSHARP_MASK_FILTER_H
