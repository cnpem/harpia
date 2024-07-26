#ifndef MORPH_GRAYSCALE_OPS_H
#define MORPH_GRAYSCALE_OPS_H

template<typename dtype>
void erosion_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void dilation_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void closing_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void opening_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose);

#endif // MORPH_GRAYSCALE_OPS_H