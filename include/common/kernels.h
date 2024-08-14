#ifndef KERNELS_H
#define KERNELS_H

#include<iostream>
#include<cuda_runtime.h>
#include<cmath>

template<typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int xsize, int ysize, int kx, int ky)
{

    int input_col;
    int input_row;

    float accumulation = 0;
    
    for(int m = 0; m < kx; m++)
    {

        for (int n = 0; n < ky; n++)
        {
            //this is needed to compute everything with respect to the center of the kernel.
            input_row = i - kx / 2 + m;
            input_col = j - ky / 2 + n;

            // Check if input_row and input_col are within bounds
            if (input_row >= 0 && input_row < xsize && input_col >= 0 && input_col < ysize)
            {
                accumulation += image[input_row * ysize + input_col];
            }
            
            //make a padding function to substitute this line of code.
            else
            {

                // Reflect padding
                if (input_row < 0)
                    input_row = -input_row;
                else if (input_row >= xsize)
                    input_row = 2 * xsize - input_row - 1;

                if (input_col < 0)
                    input_col = -input_col;
                else if (input_col >= ysize)
                    input_col = 2 * ysize - input_col - 1;

                accumulation += image[input_row * ysize + input_col];

            }
            
        }

    }

    *mean = accumulation/(kx*ky);

}

template<typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean,
                          int i, int j, int k, 
                          int xsize, int ysize, int zsize,
                          int kx, int ky, int kz)
{

    float accumulation = 0;

    int input_col;
    int input_row;
    int input_zsize;
    
    for (int l = 0; l < kz; l++)
    {

        for(int m = 0; m < kx; m++)
        {

            for (int n = 0; n < ky; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                input_row = i - kx / 2 + m;
                input_col = j - ky / 2 + n;
                input_zsize = k - kz / 2 + l;

                if (input_row >= 0 && input_row < xsize && input_col >= 0 && input_col < ysize && input_zsize >= 0 && input_zsize < zsize)
                {
                    accumulation += image[(input_zsize * xsize * ysize) + (input_row * ysize) + input_col];
                }
                
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (input_row < 0)
                    {
                        input_row = -input_row;
                    }

                    else if (input_row >= xsize)
                    {
                        input_row = 2 * xsize - input_row - 1;
                    }


                    if (input_col < 0)
                    {
                        input_col = -input_col;
                    }

                    else if (input_col >= ysize)
                    {
                        input_col = 2 * ysize - input_col - 1;
                    }

                    if (input_zsize < 0)
                    {
                        input_zsize = - input_zsize;
                    }

                    else if (input_zsize>=zsize)
                    {
                        input_zsize = 2 * zsize - input_zsize -1;
                    }
                    

                    accumulation += image[(input_zsize * xsize * ysize) + (input_row * ysize) + input_col];

                }
                
            }

        }

    }

    *mean = accumulation/(kx*ky*kz);

}


template<typename dtype>
__device__ void get_std_kernel_2d(dtype* image, float mean, float* standard_deviation, int i, int j, int xsize, int ysize, int kx, int ky)
{

    int input_col;
    int input_row;

    float accumulation = 0;
    
    for(int m = 0; m < kx; m++)
    {

        for (int n = 0; n < ky; n++)
        {
            //this is needed to compute everything with respect to the center of the kernel.
            input_row = i - kx / 2 + m;
            input_col = j - ky / 2 + n;

            // Check if input_row and input_col are within bounds
            if (input_row >= 0 && input_row < xsize && input_col >= 0 && input_col < ysize)
            {
                accumulation += pow(image[input_row * ysize + input_col] - mean,2);
            }
            
            //make a padding function to substitute this line of code.
            else
            {

                // Reflect padding
                if (input_row < 0)
                    input_row = -input_row;
                else if (input_row >= xsize)
                    input_row = 2 * xsize - input_row - 1;

                if (input_col < 0)
                    input_col = -input_col;
                else if (input_col >= ysize)
                    input_col = 2 * ysize - input_col - 1;

                accumulation += pow(image[input_row * ysize + input_col] - mean,2);

            }
            
        }

    }

    *standard_deviation = sqrt(accumulation/(kx*ky) );

}



template<typename dtype>
__device__ void get_std_kernel_3d(dtype* image, float mean, float* standard_deviation,
                          int i, int j, int k, 
                          int xsize, int ysize, int zsize,
                          int kx, int ky, int kz)
{

    float accumulation = 0;

    int input_col;
    int input_row;
    int input_zsize;
    
    for (int l = 0; l < kz; l++)
    {

        for(int m = 0; m < kx; m++)
        {

            for (int n = 0; n < ky; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                input_row = i - kx / 2 + m;
                input_col = j - ky / 2 + n;
                input_zsize = k - kz / 2 + l;

                if (input_row >= 0 && input_row < xsize && input_col >= 0 && input_col < ysize && input_zsize >= 0 && input_zsize < zsize)
                {
                    accumulation += pow(image[(input_zsize * xsize * ysize) + (input_row * ysize) + input_col] - mean,2);
                }
                
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (input_row < 0)
                    {
                        input_row = -input_row;
                    }

                    else if (input_row >= xsize)
                    {
                        input_row = 2 * xsize - input_row - 1;
                    }


                    if (input_col < 0)
                    {
                        input_col = -input_col;
                    }

                    else if (input_col >= ysize)
                    {
                        input_col = 2 * ysize - input_col - 1;
                    }

                    if (input_zsize < 0)
                    {
                        input_zsize = - input_zsize;
                    }

                    else if (input_zsize>=zsize)
                    {
                        input_zsize = 2 * zsize - input_zsize -1;
                    }
                    

                    accumulation += pow(image[(input_zsize * xsize * ysize) + (input_row * ysize) + input_col]-mean,2);

                }
                
            }

        }

    }

    *standard_deviation = sqrt(accumulation/(kx*ky*kz));

}


static void get_gaussian_kernel_2d(float** kernel, int xsize, int ysize, float sigma)
{
    /*

        kernel is given by the gaussian distribution:

        y = exp(||xi-xj||^2 /sigma^2)

    */

   //kernel allocation
    *kernel = (float*)malloc(sizeof(float)*xsize*ysize);

    if (! *kernel)
    {
        return;
    }

    int x;
    int y;

    int center_row = xsize / 2;
    int center_col = ysize / 2;

    float distance = 0;
    float normalization = 0;

    // Generate the kernel values.
    for (int i = 0; i < xsize; i++)
    {

        for (int j = 0; j < ysize; j++)
        {

            x = i - center_row;
            y = j - center_col;

            distance = x * x + y * y;

            (*kernel)[i * ysize + j] = exp(-distance / (2 * sigma * sigma + 1E-16))*1E2;
            normalization += (*kernel)[i * ysize +j];

            //std::cout<<(*kernel)[i*ysize+j]<<" ";

        }

        //std::cout<<"\n";


    }

    for (int i = 0; i < xsize*ysize; i++)
    {
        (*kernel)[i] = (*kernel)[i]/normalization;
    }
    
    
    
}


static void get_gaussian_kernel_3d(float** kernel, int xsize, int ysize, int zsize, float sigma)
{
    /*

        kernel is given by the gaussian distribution:

        y = exp(||xi-xj||^2 /sigma^2)

    */

   //kernel allocation
    *kernel = (float*)malloc(sizeof(float)*xsize*ysize*zsize);

    if (! *kernel)
    {
        return;
    }

    int x;
    int y;
    int z;

    int center_row = xsize / 2;
    int center_col = ysize / 2;
    int center_zsize = zsize / 2;

    float distance = 0;
    float normalization = 0;

    // Generate the kernel values.
    for (int k = 0; k < zsize; k++)
    {
    
        for (int i = 0; i < xsize; i++)
        {

            for (int j = 0; j < ysize; j++)
            {

                x = i - center_row;
                y = j - center_col;
                z = k - center_zsize;

                distance = x * x + y * y + z * z;

                (*kernel)[k*xsize*ysize + i * ysize + j] = exp(-distance / (2 * sigma * sigma + 1E-16))*1E2;
                normalization += (*kernel)[k*xsize*ysize + i * ysize + j];

                //std::cout<<(*kernel)[i*ysize+j]<<" ";

            }

            //std::cout<<"\n";


        }
    
    }


    for (int i = 0; i < xsize*ysize*zsize; i++)
    {
        (*kernel)[i] = (*kernel)[i]/normalization;
    }
      
}
#endif // KERNELS_H