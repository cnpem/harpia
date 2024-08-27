#ifndef STRUCTURING_ELEMENT_H
#define STRUCTURING_ELEMENT_H

// Create a structuring element of any rectangular size given
void get_structuring_element_3D(int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize);

void custum_kernel_3D(int** kernel);

void horizontal_line_kernel(int* kernel);

void vertical_line_kernel(int* kernel);

#endif  // STRUCTURING_ELEMENT_H