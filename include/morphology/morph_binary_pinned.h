#ifndef BINARY_MORPHOLOGY_PINNED_H
#define BINARY_MORPHOLOGY_PINNED_H

#include "morphology.h"

template <typename dtype>
void morph_binary_pinned(dtype* deviceImage, dtype* deviceOutput, int* kernel, int kernel_xsize,
                         int kernel_ysize, int kernel_zsize, const int xsize, const int ysize,
                         const int zsize, MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on the device
template <typename dtype>
void morph_binary_pinned_on_device(dtype* hostImage, dtype* hostOutput, int* kernel,
                                   int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on host
template <typename dtype>
void morph_binary_pinned_on_host(dtype* hostImage, dtype* hostOutput, int* kernel, int kernel_xsize,
                                 int kernel_ysize, int kernel_zsize, const int xsize,
                                 const int ysize, const int zsize, MorphOp operation);

#endif  // BINARY_MORPHOLOGY_PINNED_H