#ifndef BINARY_MORPHOLOGY_CHAIN_H
#define BINARY_MORPHOLOGY_CHAIN_H

#include "morphology.h"

template <typename dtype>
void morph_chain_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                  const int ysize, const int zsize, const int padding_bottom,
                                  const int padding_top, int* kernel, int kernel_xsize,
                                  int kernel_ysize, int kernel_zsize, MorphChain chain,
                                  const int flag_verbose);

template <typename dtype>
void morph_chain_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                int kernel_ysize, int kernel_zsize, MorphChain chain);

#endif  // BINARY_MORPHOLOGY_CHAIN_H