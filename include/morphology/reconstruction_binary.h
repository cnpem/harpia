#ifndef RECONSTRUCTION_BINARY_H
#define RECONSTRUCTION_BINARY_H

#include "morphology.h"

template <typename dtype>
void reconstruction_binary(dtype* deviceMarker, dtype* deviceMask, dtype* deviceOutput,
                           const int xsize, const int ysize, const int zsize, MorphOp operation,
                           const int flag_verbose);

template <typename dtype>
void reconstruction_binary_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                     const int xsize, const int ysize, const int zsize,
                                     MorphOp operation, const int flag_verbose);

template <typename dtype>
void reconstruction_binary_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation);

#endif  // RECONSTRUCTION_BINARY_H