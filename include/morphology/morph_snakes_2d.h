#ifndef MORPH_SNAKES_2D_H
#define MORPH_SNAKES_2D_H

#include "morphology.h"

void morph_geodesic_active_contour(float* hostImage, bool* initLs, int* iterations, float* threshold, float* balloonForce, int* smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose);

void morph_chan_vese(float* hostImage, bool* initLs, int* iterations, float* lambda1, float* lambda2, int* smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose);

