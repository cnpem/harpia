#ifndef COMPARE_ARRAYS_GRAYSCALE_H
#define COMPARE_ARRAYS_GRAYSCALE_H

#include "morphology.h"

template <typename dtype>
void compare_arrays_grayscale(dtype* deviceImage1, dtype* deviceImage2, int* deviceOutput,
                              const int size, const int flag_verbose);

template <typename dtype>
void compare_arrays_grayscale_on_device(dtype* hostImage1, dtype* hostImage2, int* hostOutput,
                                        const int size, const int flag_verbose);

template <typename dtype>
void compare_arrays_grayscale_on_host(dtype* hostImage1, dtype* hostImage2, int* hostOutput,
                                      const int size);

#endif  // COMPARE_ARRAYS_GRAYSCALE_H