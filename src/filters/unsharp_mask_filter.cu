#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/unsharp_mask_filter.h"


/*
    adapted from: https://www.nv5geospatialsoftware.com/docs/unsharp_mask.html

    temp = Image - Convol ( Image, Gaussian )
    output = Image + A * temp * ( |temp| ≥ T )
*/


template<typename dtype>
void unsharp_mask_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma, float ammount, float threshold, bool type)
{
    //gaussian filter application.
    gaussian_filtering(image, output, xsize, ysize, zsize, sigma,type);

    for (int idx = 0; idx < xsize; ++idx)
    {

        for (int idy = 0; idy < ysize; ++idy)
        {

            for (int idz = 0; idz < zsize; ++idz)
            {

                float temp;

                temp = image[idz * xsize * ysize +  idx * ysize + idy] - output[idz * xsize * ysize +  idx * ysize + idy];

                if (abs(temp) >= threshold)
                {
                    output[idz * xsize * ysize +  idx * ysize + idy] = (float)(image[idz * xsize * ysize +  idx * ysize + idy] + ammount * temp);
                }

                else
                {
                    output[idz * xsize * ysize +  idx * ysize + idy] = (float)image[idz * xsize * ysize +  idx * ysize + idy];
                }

            }

        }

    }
    
}

// Explicit instantiation
template void unsharp_mask_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize, float sigma, float ammount, float threshold, bool type);
template void unsharp_mask_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize, float sigma, float ammount, float threshold, bool type);
template void unsharp_mask_filtering<unsigned int>(unsigned int* image, float* output, int xsize, int ysize, int zsize, float sigma, float ammount, float threshold, bool type);
