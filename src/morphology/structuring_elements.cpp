#include <stdio.h>
#include <stdlib.h>

#include "../../include/morphology/structuring_elements.h"

// Create a structuring element of any rectangular size given
void get_structuring_element_3D(int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize)
{
    //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

    /*
       +---------+
      / 1  1  1 /|
     / 1  1  1 / |
    / 1  1  1 /  |
   +---------+   |
   | 1  1  1 |   |
   | 1  1  1 |   +
   | 1  1  1 |  / 
   |         | /
   +---------+/ kernel_ysize(rows) x kernel_xsize(columns) x kernel_zsize(channels/slices)
    
    */

    int size =  kernel_xsize*kernel_ysize*kernel_zsize;

    if (!kernel)
    {
        return;
    }

    for (int i=0; i<size; i++){
        kernel[i] = 1;
    }
    
}

void horizontal_line_kernel(int* kernel)
{
    if (!kernel)
    {   
        printf("Failed to define kernel.\n");
        return;
    }

    int *ik = kernel;
    for(int i=0; i<3; i++)
    {
        ik[0] = 1;
        ik[1] = 1;
        ik[2] = 1;

        ik[3] = 0;
        ik[4] = 0;
        ik[5] = 0;

        ik[6] = 0;
        ik[7] = 0;
        ik[8] = 0;

        ik+= 9; //repeat the defined slice along z axis
    }
}

void vertical_line_kernel(int* kernel)
{
    if (!kernel)
    {   
        printf("Failed to define kernel.\n");
        return;
    }
    
    int *ik = kernel;
    for(int i=0; i<3; i++)
    {
        ik[0] = 1;
        ik[1] = 0;
        ik[2] = 0;

        ik[3] = 1;
        ik[4] = 0;
        ik[5] = 0;

        ik[6] = 1;
        ik[7] = 0;
        ik[8] = 0;

        ik+= 9; //repeat the defined slice along z axis
    }
}

void custum_kernel_3D(int* kernel)
{
    //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

    /*
        +--------------+
        |   1   1   1  |
        |  -1  -1  -1  |
        |  -1  -1  -1  |
        +--------------+
    
    */

    if (!kernel)
    {
        return;
    }

    for(int i=0; i<3; i++)
    {
        kernel[0] = 1;
        kernel[1] = -1;
        kernel[2] = -1;

        kernel[3] = 1;
        kernel[4] = -1;
        kernel[5] = -1;

        kernel[6] = 1;
        kernel[7] = -1;
        kernel[8] = -1;

        kernel+= 9; //repeat the defined slice along z axis
    }
// for the grayscale erosion, it only makes sense for the kernel to have values 1's ore -1's (don't care)
}