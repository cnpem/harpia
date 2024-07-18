#ifndef GRAYSCALE_MORPHOLOGY_CHAIN_H
#define GRAYSCALE_MORPHOLOGY_CHAIN_H

#include "morphology.h"

template<typename dtype>
void morphChainGrayscaleOnDevice(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                            const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                            MorphChain chain, const int flag_verbose);

template<typename dtype>
void morphChainGrayscaleOnHost(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                            const int xsize, const int ysize, const int zsize, MorphChain chain);

#endif // GRAYSCALE_MORPHOLOGY_CHAIN_H