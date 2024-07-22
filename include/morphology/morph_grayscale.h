#ifndef GRAYSCALE_MORPHOLOGY_H
#define GRAYSCALE_MORPHOLOGY_H

#include "morphology.h"

template<typename dtype>
void morphGrayscaleOnDevice(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, 
                            int kernel_zsize, const int xsize, const int ysize, const int zsize, MorphOp operation, const int flag_verbose);

template<typename dtype>
void morphGrayscaleOnHost(dtype *hostImage, dtype *hostOutput, 
             int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
             const int xsize, const int ysize, const int zsize, MorphOp operation);


template<typename dtype>
CUDA_GLOBAL
void morphGrayscaleKernel(dtype *deviceImage, dtype *deviceOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                          const int xsize, const int ysize, const int zsize, MorphOp operation);
#endif // GRAYSCALE_MORPHOLOGY_H