#ifndef BINARY_MORPHOLOGY_H
#define BINARY_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void morph_binary(dtype* deviceImage, dtype* deviceOutput, const int xsize, const int ysize,
                  const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                  int kernel_zsize, MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on the device
template <typename dtype>
void morph_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                            int kernel_zsize, MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on host
template <typename dtype>
void morph_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                          int kernel_zsize, MorphOp operation);

#endif  // BINARY_MORPHOLOGY_H