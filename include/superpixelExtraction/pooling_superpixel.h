#ifndef POOLING_SUPERPIXEL_H
#define POOLING_SUPERPIXEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void superpixel_feature_extract(
    float* hostImage,
    int* hostSuperPixel,
    float* hostOutput,
    int xsize, int ysize, int zsize,
    int nsuperpixels,
    int nfeatures,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern,
    bool output_mean = true,
    bool output_min = false,
    bool output_max = false,
    int verbose = 0);

#endif