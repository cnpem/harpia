#ifndef DEVICE_FEAT_EXTRACT_H
#define DEVICE_FEAT_EXTRACT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void DeviceFeatExtraction2D(float* hostImage, float* hostOutput,
    int xsize, int ysize, int zsize,
    int nfeatures,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern,
    int verbose, 
    float gpuMemory, 
    int ngpus);

#endif // DEVICE_FEAT_EXTRACT_H