#ifndef GEODESIC_GRAYSCALE_MORPHOLOGY_H
#define GEODESIC_GRAYSCALE_MORPHOLOGY_H

#include "morphology.h"

template <typename dtype>
void geodesic_morph_grayscale(dtype* deviceImage, dtype* deviceMask, dtype* deviceOutput,
                              const int xsize, const int ysize, const int zsize,
                              const int flag_verbose, MorphOp operation);

template <typename dtype>
void geodesic_morph_grayscale_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                        const int xsize, const int ysize, const int zsize,
                                        const int flag_verbose, MorphOp operation);

template <typename dtype>
void geodesic_morph_grayscale_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                      const int xsize, const int ysize, const int zsize,
                                      MorphOp operation);

#endif  // GEODESIC_GRAYSCALE_MORPHOLOGY_H