#ifndef MORPH_SNAKES_2D_H
#define MORPH_SNAKES_2D_H

void morph_geodesic_active_contour(float* hostImage, bool* initLs, const int iterations, const float balloonForce, const float threshold, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose);

void morph_chan_vese(float* hostImage, bool* initLs, const int iterations, const float lambda1, const float lambda2, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose);

#endif  // MORPH_SNAKES_2D_H