#ifndef GRAYSCALE_MORPHOLOGY_PINNED_H
#define GRAYSCALE_MORPHOLOGY_PINNED_H

#include "morphology.h"

template <typename dtype>
void morph_grayscale_pinned(dtype* deviceImage, dtype* deviceOutput, const int xsize,
                            const int ysize, const int zsize, int* deviceKernel, int kernel_xsize,
                            int kernel_ysize, int kernel_zsize, MorphOp operation,
                            const int flag_verbose);

template <typename dtype>
void morph_grayscale_pinned_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                      const int ysize, const int zsize, int* kernel,
                                      int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                      MorphOp operation, const int flag_verbose);

template <typename dtype>
void morph_grayscale_pinned_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                    const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                    int kernel_ysize, int kernel_zsize, MorphOp operation);

#endif  // GRAYSCALE_MORPHOLOGY_PINNED_H