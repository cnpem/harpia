#ifndef GRAYSCALE_MORPHOLOGY_H
#define GRAYSCALE_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void morph_grayscale(dtype* deviceImage, dtype* deviceOutput, int* deviceKernel, int kernel_xsize,
                     int kernel_ysize, int kernel_zsize, const int xsize, const int ysize,
                     const int zsize, MorphOp operation, const int flag_verbose);

template <typename dtype>
void morph_grayscale_on_device(dtype* hostImage, dtype* hostOutput, int* kernel, int kernel_xsize,
                               int kernel_ysize, int kernel_zsize, const int xsize, const int ysize,
                               const int zsize, MorphOp operation, const int flag_verbose);

template <typename dtype>
void morph_grayscale_on_host(dtype* hostImage, dtype* hostOutput, int* kernel, int kernel_xsize,
                             int kernel_ysize, int kernel_zsize, const int xsize, const int ysize,
                             const int zsize, MorphOp operation);

#endif  // GRAYSCALE_MORPHOLOGY_H