#ifndef BINARY_MORPHOLOGY_CHAIN_H
#define BINARY_MORPHOLOGY_CHAIN_H

#include "morphology.h"

template<typename dtype>
void morph_chain_binary_on_device(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                            const int xsize, const int ysize, const int zsize, MorphChain chain, const int flag_verbose);

template<typename dtype>
void morph_chain_binary_on_host(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                            const int xsize, const int ysize, const int zsize, MorphChain chain);

#endif // BINARY_MORPHOLOGY_CHAIN_H