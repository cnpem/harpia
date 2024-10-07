#ifndef GEODESIC_BINARY_MORPHOLOGY_H
#define GEODESIC_BINARY_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void geodesic_morph_binary(dtype* deviceImage, dtype* deviceMask, dtype* deviceOutput,
                           const int xsize, const int ysize, const int zsize,
                           const int flag_verbose, const int padding_bottom, const int padding_top,
                           MorphOp operation);

// Slide kernel and morphological operation over all input image pixels on the device
template <typename dtype>
void geodesic_morph_binary_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                     const int xsize, const int ysize, const int zsize,
                                     const int flag_verbose, const int padding_bottom,
                                     const int padding_top, MorphOp operation);

// Slide kernel and morphological operation over all input image pixels on host
template <typename dtype>
void geodesic_morph_binary_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                   const int xsize, const int ysize, const int zsize,
                                   MorphOp operation);

#endif  // GEODESIC_BINARY_MORPHOLOGY_H