#ifndef NLMEANS_FILTER_H
#define NLMEANS_FILTER_H

#include <cuda.h>
#include <cuda_runtime.h>

// Template device function for calculating the non-local means kernel
template <typename dtype>
__device__ void get_nlmean_kernel_2d(dtype* image, double* mean, int idx, int idy,
                                     int xsize, int ysize, int small_window, int big_window,
                                     double h, double sigma);

// Template kernel function for the non-local means filter
template <typename dtype>
__global__ void nlmeans_filter_kernel_2d(dtype* deviceImage, double* deviceOutput,
                                         int xsize, int ysize, int small_window,
                                         int big_window, double h, double sigma);

// Template function for non-local means filtering
template <typename dtype>
void nlmeans_filtering(dtype* hostImage, double* hostOutput, int xsize, int ysize,
                       int small_window, int big_window, double h, double sigma);

#endif // NLMEANS_FILTER_H
