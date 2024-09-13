#ifndef MORPH_binary_on_device_OPS_H
#define MORPH_binary_on_device_OPS_H

template <typename dtype>
void erosion_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                              const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                              int kernel_zsize, const int flag_verbose);

template <typename dtype>
void erosion_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                            int kernel_zsize);

template <typename dtype>
void dilation_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, int* kernel, int kernel_xsize,
                               int kernel_ysize, int kernel_zsize, const int flag_verbose);

template <typename dtype>
void dilation_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize);

template <typename dtype>
void closing_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                              const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                              int kernel_zsize, const int flag_verbose);

template <typename dtype>
void closing_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                            int kernel_zsize);

template <typename dtype>
void opening_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                              const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                              int kernel_zsize, const int flag_verbose);

template <typename dtype>
void opening_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                            int kernel_zsize);

template <typename dtype>
void geodesic_erosion_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                       const int ysize, const int zsize, dtype* hostMask,
                                       const int flag_verbose);

template <typename dtype>
void geodesic_erosion_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                     const int ysize, const int zsize, dtype* hostMask);

template <typename dtype>
void geodesic_dilation_binary_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                        const int ysize, const int zsize, dtype* hostMask,
                                        const int flag_verbose);

template <typename dtype>
void geodesic_dilation_binary_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                      const int ysize, const int zsize, dtype* hostMask);

#endif  // MORPH_binary_on_device_OPS_H