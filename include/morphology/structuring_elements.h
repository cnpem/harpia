#ifndef STRUCTURING_ELEMENT_H
#define STRUCTURING_ELEMENT_H

/**
 * @brief Creates a 3D structuring element with all elements set to 1.
 * 
 * @param kernel Pointer to the kernel array.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 */
void get_structuring_element_3D(int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize);

/**
 * @brief Creates a 3D horizontal line structuring element.
 * 
 * @param kernel Pointer to the kernel array.
 */
void horizontal_line_kernel(int* kernel);

/**
 * @brief Creates a 3D vertical line structuring element.
 * 
 * @param kernel Pointer to the kernel array.
 */
void vertical_line_kernel(int* kernel);

/**
 * @brief Creates a custom 3D structuring element with a specific pattern.
 * 
 * @param kernel Pointer to the kernel array.
 */
 void custum_kernel_3D(int** kernel);

#endif  // STRUCTURING_ELEMENT_H