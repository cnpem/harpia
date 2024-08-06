#ifndef MORPH_BINARY_OPS_H
#define MORPH_BINARY_OPS_H

template<typename dtype>
void erosion_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void dilation_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void closing_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void opening_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void geodesic_erosion_binary(dtype *hostImage, dtype *hostOutput, dtype *hostMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void geodesic_dilation_binary(dtype *hostImage, dtype *hostOutput, dtype *hostMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

#endif // MORPH_BINARY_OPS_H