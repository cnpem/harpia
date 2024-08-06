#ifndef GEODESIC_BINARY_MORPHOLOGY_H
#define GEODESIC_BINARY_MORPHOLOGY_H

#include "morphology.h"

// Slide kernel and morphological operation over all input image pixels on the device
template<typename dtype>
void geodesic_morph_binary_on_device(dtype *hostImage, dtype *hostOutput, dtype *hostMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                                     const int xsize, const int ysize, const int zsize, MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on host
template<typename dtype>
void geodesic_morph_binary_on_host(dtype *hostImage, dtype *hostOutput, dtype *hostMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                                   const int xsize, const int ysize, const int zsize, MorphOp operation);

template<typename dtype>
CUDA_GLOBAL
void geodesic_morph_binary(dtype *deviceImage, dtype *deviceOutput, dtype *deviceMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                           const int xsize, const int ysize, const int zsize, MorphOp operation);

#endif // GEODESIC_BINARY_MORPHOLOGY_H