#ifndef FILTERS_IN_DEVICE_H
#define FILTERS_IN_DEVICE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void applyGaussianFilterDevice2D(
    float* d_image,
    float* d_image_smoothed,
    float sigma,
    int xsize, int ysize, int zsize);


void applyPrewittFilterDevice2D(
    float* d_image_smoothed,
    float* d_temp_image,
    int xsize, int ysize, int zsize);


void applyLocalBinaryPatternDevice2D(
    float* d_image_smoothed,
    float* d_temp_image,
    int xsize, int ysize, int zsize);

void applyHessianEigenvaluesDevice2D(float* d_image, float* d_eigen1, float* d_eigen2,
                        int xsize, int ysize, int zsize, int step);

void applyShapeIndexDevice2D(float* d_image, float* d_shape_index, 
                int xsize, int ysize, int zsize, int step);

#endif // FILTERS_IN_DEVICE_H