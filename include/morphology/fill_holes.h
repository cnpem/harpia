#ifndef FILL_HOLES_H
#define FILL_HOLES_H

#include "morphology.h"

template<typename dtype>
void fill_holes_on_device(dtype *hostImage, dtype *hostOutput, const int xsize, const int ysize, const int zsize, 
                         const int flag_verbose);

template<typename dtype>
void fill_holes_on_host(dtype *hostImage, dtype *hostOutput, const int xsize, const int ysize, const int zsize);

#endif // FILL_HOLES_H