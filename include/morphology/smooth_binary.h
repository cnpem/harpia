#ifndef BINARY_SMOOTH_H
#define BINARY_SMOOTH_H

#include "morphology.h"

template <typename dtype>
void smooth_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, const int flag_verbose, const int padding_bottom,
                             const int padding_top, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize);

template <typename dtype>
void smooth_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                           int kernel_zsize);

#endif  // BINARY_SMOOTH_H