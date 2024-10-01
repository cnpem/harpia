#ifndef GRAYSCALE_MORPHOLOGY_CHAIN_H
#define GRAYSCALE_MORPHOLOGY_CHAIN_H

#include "morphology.h"

template <typename dtype>
void morph_chain_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                     const int ysize, const int zsize, const int flag_verbose,
                                     const int padding_bottom, const int padding_top, int* kernel,
                                     int kernel_xsize, int kernel_ysize, int kernel_zsize,
                                     MorphChain chain);

template <typename dtype>
void morph_chain_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                   const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                   int kernel_ysize, int kernel_zsize, MorphChain chain);

#endif  // GRAYSCALE_MORPHOLOGY_CHAIN_H