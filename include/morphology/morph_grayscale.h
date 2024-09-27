#ifndef GRAYSCALE_MORPHOLOGY_H
#define GRAYSCALE_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void morph_grayscale(dtype* deviceImage, dtype* deviceOutput, const int xsize, const int ysize,
                     const int zsize, const int flag_verbose, const int padding_bottom,
                     const int padding_top, int* deviceKernel, int kernel_xsize, int kernel_ysize,
                     int kernel_zsize, MorphOp operation);

template <typename dtype>
void morph_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, const int flag_verbose,
                               const int padding_bottom, const int padding_top, int* kernel,
                               int kernel_xsize, int kernel_ysize, int kernel_zsize,
                               MorphOp operation);

template <typename dtype>
void morph_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize, MorphOp operation);

#endif  // GRAYSCALE_MORPHOLOGY_H