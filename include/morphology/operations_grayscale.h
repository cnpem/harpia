#ifndef MORPH_grayscale_on_device_OPS_H
#define MORPH_grayscale_on_device_OPS_H

template <typename dtype>
void erosion_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, bool gpu);

template <typename dtype>
void dilation_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                        int kernel_ysize, int kernel_zsize, bool gpu);

template <typename dtype>
void closing_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, bool gpu);

template <typename dtype>
void opening_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, bool gpu);

template <typename dtype>
void bottom_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                int kernel_ysize, int kernel_zsize, bool gpu);

template <typename dtype>
void top_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize, const int zsize,
             const int flag_verbose, int* kernel, int kernel_xsize, int kernel_ysize,
             int kernel_zsize, bool gpu);

#endif  // MORPH_grayscale_on_device_OPS_H