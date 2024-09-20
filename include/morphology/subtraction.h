#ifndef SUBTRACTION_H
#define SUBTRACTION_H

#include "morphology.h"

//kernel for image subtraction
template <typename dtype>
void subtraction(dtype* deviceImage1, dtype* deviceImage2, dtype* deviceOutput, const int size,
                 const int flag_verbose);

// Slide kernel and erosion operation over all input image pixels
template <typename dtype>
void subtraction_on_device(dtype* hostImage1, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, dtype* hostImage2, const int flag_verbose);

// Slide kernel and erosion operation over all input image pixels
template <typename dtype>
void subtraction_on_host(dtype* hostImage1, dtype* hostImage2, dtype* hostOutput, const int size);

#endif  // SUBTRACTION_H