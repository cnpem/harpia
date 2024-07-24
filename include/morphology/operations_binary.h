#ifndef MORPH_BINARY_OPS_H
#define MORPH_BINARY_OPS_H

template<typename dtype>
void erosionBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void dilationBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void closingBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void openingBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose);

#endif // MORPH_BINARY_OPS_H