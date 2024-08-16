#include <stdio.h>
#include <stdlib.h>
#include "../../include/morphology/structuring_elements.h"

/**
 * @brief Creates a 3D structuring element with all elements set to 1.
 * 
 * @param kernel Pointer to the kernel array.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 */
void get_structuring_element_3D(int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize)
{
    int size = kernel_xsize * kernel_ysize * kernel_zsize;

    if (!kernel)
    {
        return;
    }

    for (int i = 0; i < size; i++){
        kernel[i] = 1;
    }
}

/**
 * @brief Creates a 3D horizontal line structuring element.
 * 
 * @param kernel Pointer to the kernel array.
 */
void horizontal_line_kernel(int* kernel)
{
    if (!kernel)
    {   
        printf("Failed to define kernel.\n");
        return;
    }

    int *ik = kernel;
    for (int i = 0; i < 3; i++)
    {
        ik[0] = 1; ik[1] = 1; ik[2] = 1;
        ik[3] = 0; ik[4] = 0; ik[5] = 0;
        ik[6] = 0; ik[7] = 0; ik[8] = 0;

        ik += 9; // Repeat the defined slice along z axis
    }
}

/**
 * @brief Creates a 3D vertical line structuring element.
 * 
 * @param kernel Pointer to the kernel array.
 */
void vertical_line_kernel(int* kernel)
{
    if (!kernel)
    {   
        printf("Failed to define kernel.\n");
        return;
    }

    int *ik = kernel;
    for (int i = 0; i < 3; i++)
    {
        ik[0] = 1; ik[1] = 0; ik[2] = 0;
        ik[3] = 1; ik[4] = 0; ik[5] = 0;
        ik[6] = 1; ik[7] = 0; ik[8] = 0;

        ik += 9; // Repeat the defined slice along z axis
    }
}

/**
 * @brief Creates a custom 3D structuring element with a specific pattern.
 * 
 * @param kernel Pointer to the kernel array.
 */
void custum_kernel_3D(int* kernel)
{
    if (!kernel)
    {
        return;
    }

    for (int i = 0; i < 3; i++)
    {
        kernel[0] = 1; kernel[1] = -1; kernel[2] = -1;
        kernel[3] = 1; kernel[4] = -1; kernel[5] = -1;
        kernel[6] = 1; kernel[7] = -1; kernel[8] = -1;

        kernel += 9; // Repeat the defined slice along z axis
    }
}