#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include "../gaussian/gaussian_filter.h"
#include "unsharp_mask_filter.h"


/*
    adapted from: https://www.nv5geospatialsoftware.com/docs/unsharp_mask.html

    temp = Image - Convol ( Image, Gaussian )
    output = Image + A * temp * ( |temp| ≥ T )
*/


template<typename dtype>
void unsharp_mask_filtering(dtype* image, float* output, int rows, int cols, int depth, float sigma, float ammount, float threshold, bool type)
{
    //gaussian filter application.
    gaussian_filtering(image, output, rows, cols, depth, sigma,type);

    for (int idx = 0; idx < rows; ++idx)
    {

        for (int idy = 0; idy < cols; ++idy)
        {

            for (int idz = 0; idz < depth; ++idz)
            {

                float temp;

                temp = image[idz * rows * cols +  idx * cols + idy] - output[idz * rows * cols +  idx * cols + idy];

                if (abs(temp) >= threshold)
                {
                    output[idz * rows * cols +  idx * cols + idy] = (float)(image[idz * rows * cols +  idx * cols + idy] + ammount * temp);
                }

                else
                {
                    output[idz * rows * cols +  idx * cols + idy] = (float)image[idz * rows * cols +  idx * cols + idy];
                }

            }

        }

    }
    
}

// Explicit instantiation for float
template void unsharp_mask_filtering<float>(float* image, float* output, int rows, int cols, int depth, float sigma, float ammount, float threshold, bool type);
