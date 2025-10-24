#ifndef WATERSHED_LINES_H
#define WATERSHED_LINES_H

#include <cuda_runtime.h>
#include <iostream>

// Host wrapper function declaration
template <typename in_dtype>
void compute_boundaries2d(in_dtype* labels, int* boundaries, int xsize, int ysize);


#endif // WATERSHED_LINES_H
