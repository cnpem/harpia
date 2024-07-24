#ifndef BOTTOM_HAT_H
#define BOTTOM_HAT_H

template<typename dtype>
void bottomHat(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);

template<typename dtype>
void bottomHatOnHost(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose);
#endif // BOTTOM_HAT_H