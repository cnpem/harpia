#ifndef POOLING_SUPERPIXEL_H
#define POOLING_SUPERPIXEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void DeviceSuperpixelPooling2D(float* hostImage,
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
    bool output_mean,
    bool output_min,
    bool output_max,
    int flag_verbose, 
    float gpuMemory, 
    int ngpus);

#endif