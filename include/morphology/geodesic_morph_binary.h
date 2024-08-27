#ifndef GEODESIC_BINARY_MORPHOLOGY_H
#define GEODESIC_BINARY_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void geodesic_morph_binary(dtype* deviceImage, dtype* deviceOutput, dtype* deviceMask,
                           const int xsize, const int ysize, const int zsize, MorphOp operation,
                           const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on the device
template <typename dtype>
void geodesic_morph_binary_on_device(dtype* hostImage, dtype* hostOutput, dtype* hostMask,
                                     const int xsize, const int ysize, const int zsize,
                                     MorphOp operation, const int flag_verbose);

// Slide kernel and morphological operation over all input image pixels on host
template <typename dtype>
void geodesic_morph_binary_on_host(dtype* hostImage, dtype* hostOutput, dtype* hostMask,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation);

#endif  // GEODESIC_BINARY_MORPHOLOGY_H