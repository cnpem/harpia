#ifndef BOTTOM_HAT_H
#define BOTTOM_HAT_H

template <typename dtype>
void bottom_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                          int kernel_zsize, const int flag_verbose);

template <typename dtype>
void bottom_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                        int kernel_zsize);
#endif  // BOTTOM_HAT_H