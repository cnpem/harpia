#ifndef RECONSTRUCTION_GRAYSCALE_H
#define RECONSTRUCTION_GRAYSCALE_H

#include "morphology.h"

template <typename dtype>
void reconstruction_grayscale(dtype* deviceMarker, dtype* deviceOutput, const int xsize,
                              const int ysize, const int zsize, dtype* deviceMask,
                              MorphOp operation, const int flag_verbose);

template <typename dtype>
void reconstruction_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                        const int ysize, const int zsize, dtype* hostMask,
                                        MorphOp operation, const int flag_verbose);

template <typename dtype>
void reconstruction_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                      const int ysize, const int zsize, dtype* hostMask,
                                      MorphOp operation);

#endif  // RECONSTRUCTION_GRAYSCALE_H