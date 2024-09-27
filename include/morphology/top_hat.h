#ifndef TOP_HAT_H
#define TOP_HAT_H

template <typename dtype>
void top_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize);

template <typename dtype>
void top_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                     const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                     int kernel_zsize);

template <typename dtype>
void top_hat_aviso_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                             int kernel_ysize, int kernel_zsize);

template <typename dtype>
void top_hat_aviso_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                           const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                           int kernel_zsize);

#endif  // TOP_HAT_H