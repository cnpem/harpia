#ifndef BINARY_MORPHOLOGY_H
#define BINARY_MORPHOLOGY_H

#include "morphology.h"

// Slide kernel and morphological operation over all input image pixels on the device
template<typename dtype>
void morphBinaryOnDevice(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                         const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, 
                         const int block_zsize, MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on host
template<typename dtype>
void morphBinaryOnHost(dtype *hostImage, dtype *hostOutput, 
                       int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                       const int xsize, const int ysize, const int zsize, MorphOp operation);

template<typename dtype>
CUDA_GLOBAL
void morphBinaryKernel(dtype *deviceImage, dtype *deviceOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, MorphOp operation);

#endif // BINARY_MORPHOLOGY_H