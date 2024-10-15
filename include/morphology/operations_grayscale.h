#ifndef MORPH_grayscale_on_device_OPS_H
#define MORPH_grayscale_on_device_OPS_H

#include "morphology.h"

template <typename dtype>
void erosion_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

template <typename dtype>
void dilation_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                        int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

template <typename dtype>
void closing_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

template <typename dtype>
void opening_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

template <typename dtype>
void geodesic_erosion_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                const int xsize, const int ysize, const int zsize,
                                const int flag_verbose, float gpuMemory, bool gpu);

template <typename dtype>
void geodesic_dilation_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                 const int xsize, const int ysize, const int zsize,
                                 const int flag_verbose, float gpuMemory, bool gpu);

template <typename dtype>
void reconstruction_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              MorphOp operation, bool gpu);

template <typename dtype>
void bottom_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

template <typename dtype>
void top_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize, const int zsize,
             const int flag_verbose, int* kernel, int kernel_xsize, int kernel_ysize,
             int kernel_zsize, float gpuMemory, bool gpu);

template <typename dtype>
void top_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                            int kernel_ysize, int kernel_zsize, bool gpu);

template <typename dtype>
void bottom_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, const int flag_verbose,
                               int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize,
                               bool gpu);

#endif  // MORPH_grayscale_on_device_OPS_H