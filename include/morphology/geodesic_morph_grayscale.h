#ifndef GEODESIC_GRAYSCALE_MORPHOLOGY_H
#define GEODESIC_GRAYSCALE_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void geodesic_morph_grayscale(dtype* deviceImage, dtype* deviceOutput, dtype* deviceMask,
                              const int xsize, const int ysize, const int zsize, MorphOp operation,
                              const int flag_verbose);

template <typename dtype>
void geodesic_morph_grayscale_on_device(dtype* hostImage, dtype* hostOutput, dtype* hostMask,
                                        const int xsize, const int ysize, const int zsize,
                                        MorphOp operation, const int flag_verbose);

template <typename dtype>
void geodesic_morph_grayscale_on_host(dtype* hostImage, dtype* hostOutput, dtype* hostMask,
                                      const int xsize, const int ysize, const int zsize,
                                      MorphOp operation);

#endif  // GEODESIC_GRAYSCALE_MORPHOLOGY_H