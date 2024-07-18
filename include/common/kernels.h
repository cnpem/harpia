#ifndef KERNELS_H
#define KERNELS_H

#include<iostream>
#include<cuda_runtime.h>
#include<cmath>

template<typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int i, int j, int rows, int cols, int rows_kernel, int cols_kernel)
{

    int input_col;
    int input_row;

    float accumulation = 0;
    
    for(int m = 0; m < rows_kernel; m++)
    {

        for (int n = 0; n < cols_kernel; n++)
        {
            //this is needed to compute everything with respect to the center of the kernel.
            input_row = i - rows_kernel / 2 + m;
            input_col = j - cols_kernel / 2 + n;

            // Check if input_row and input_col are within bounds
            if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols)
            {
                accumulation += image[input_row * cols + input_col];
            }
            
            //make a padding function to substitute this line of code.
            else
            {

                // Reflect padding
                if (input_row < 0)
                    input_row = -input_row;
                else if (input_row >= rows)
                    input_row = 2 * rows - input_row - 1;

                if (input_col < 0)
                    input_col = -input_col;
                else if (input_col >= cols)
                    input_col = 2 * cols - input_col - 1;

                accumulation += image[input_row * cols + input_col];

            }
            
        }

    }

    *mean = accumulation/(rows_kernel*cols_kernel);

}

template<typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean,
                          int i, int j, int k, 
                          int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel)
{

    float accumulation = 0;

    int input_col;
    int input_row;
    int input_depth;
    
    for (int l = 0; l < depth_kernel; l++)
    {

        for(int m = 0; m < rows_kernel; m++)
        {

            for (int n = 0; n < cols_kernel; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                input_row = i - rows_kernel / 2 + m;
                input_col = j - cols_kernel / 2 + n;
                input_depth = k - depth_kernel / 2 + l;

                if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols && input_depth >= 0 && input_depth < depth)
                {
                    accumulation += image[(input_depth * rows * cols) + (input_row * cols) + input_col];
                }
                
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (input_row < 0)
                    {
                        input_row = -input_row;
                    }

                    else if (input_row >= rows)
                    {
                        input_row = 2 * rows - input_row - 1;
                    }


                    if (input_col < 0)
                    {
                        input_col = -input_col;
                    }

                    else if (input_col >= cols)
                    {
                        input_col = 2 * cols - input_col - 1;
                    }

                    if (input_depth < 0)
                    {
                        input_depth = - input_depth;
                    }

                    else if (input_depth>=depth)
                    {
                        input_depth = 2 * depth - input_depth -1;
                    }
                    

                    accumulation += image[(input_depth * rows * cols) + (input_row * cols) + input_col];

                }
                
            }

        }

    }

    *mean = accumulation/(rows_kernel*cols_kernel*depth_kernel);

}


template<typename dtype>
__device__ void get_std_kernel_2d(dtype* image, float mean, float* standard_deviation, int i, int j, int rows, int cols, int rows_kernel, int cols_kernel)
{

    int input_col;
    int input_row;

    float accumulation = 0;
    
    for(int m = 0; m < rows_kernel; m++)
    {

        for (int n = 0; n < cols_kernel; n++)
        {
            //this is needed to compute everything with respect to the center of the kernel.
            input_row = i - rows_kernel / 2 + m;
            input_col = j - cols_kernel / 2 + n;

            // Check if input_row and input_col are within bounds
            if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols)
            {
                accumulation += pow(image[input_row * cols + input_col] - mean,2);
            }
            
            //make a padding function to substitute this line of code.
            else
            {

                // Reflect padding
                if (input_row < 0)
                    input_row = -input_row;
                else if (input_row >= rows)
                    input_row = 2 * rows - input_row - 1;

                if (input_col < 0)
                    input_col = -input_col;
                else if (input_col >= cols)
                    input_col = 2 * cols - input_col - 1;

                accumulation += pow(image[input_row * cols + input_col] - mean,2);

            }
            
        }

    }

    *standard_deviation = sqrt(accumulation/(rows_kernel*cols_kernel) );

}



template<typename dtype>
__device__ void get_std_kernel_3d(dtype* image, float mean, float* standard_deviation,
                          int i, int j, int k, 
                          int rows, int cols, int depth,
                          int rows_kernel, int cols_kernel, int depth_kernel)
{

    float accumulation = 0;

    int input_col;
    int input_row;
    int input_depth;
    
    for (int l = 0; l < depth_kernel; l++)
    {

        for(int m = 0; m < rows_kernel; m++)
        {

            for (int n = 0; n < cols_kernel; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                input_row = i - rows_kernel / 2 + m;
                input_col = j - cols_kernel / 2 + n;
                input_depth = k - depth_kernel / 2 + l;

                if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols && input_depth >= 0 && input_depth < depth)
                {
                    accumulation += pow(image[(input_depth * rows * cols) + (input_row * cols) + input_col] - mean,2);
                }
                
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (input_row < 0)
                    {
                        input_row = -input_row;
                    }

                    else if (input_row >= rows)
                    {
                        input_row = 2 * rows - input_row - 1;
                    }


                    if (input_col < 0)
                    {
                        input_col = -input_col;
                    }

                    else if (input_col >= cols)
                    {
                        input_col = 2 * cols - input_col - 1;
                    }

                    if (input_depth < 0)
                    {
                        input_depth = - input_depth;
                    }

                    else if (input_depth>=depth)
                    {
                        input_depth = 2 * depth - input_depth -1;
                    }
                    

                    accumulation += pow(image[(input_depth * rows * cols) + (input_row * cols) + input_col]-mean,2);

                }
                
            }

        }

    }

    *standard_deviation = sqrt(accumulation/(rows_kernel*cols_kernel*depth_kernel));

}


static void get_gaussian_kernel_2d(float** kernel, int rows, int cols, float sigma)
{
    /*

        kernel is given by the gaussian distribution:

        y = exp(||xi-xj||^2 /sigma^2)

    */

   //kernel allocation
    *kernel = (float*)malloc(sizeof(float)*rows*cols);

    if (! *kernel)
    {
        return;
    }

    int x;
    int y;

    int center_row = rows / 2;
    int center_col = cols / 2;

    float distance = 0;
    float normalization = 0;

    // Generate the kernel values.
    for (int i = 0; i < rows; i++)
    {

        for (int j = 0; j < cols; j++)
        {

            x = i - center_row;
            y = j - center_col;

            distance = x * x + y * y;

            (*kernel)[i * cols + j] = exp(-distance / (2 * sigma * sigma + 1E-16))*1E2;
            normalization += (*kernel)[i * cols +j];

            //std::cout<<(*kernel)[i*cols+j]<<" ";

        }

        //std::cout<<"\n";


    }

    for (int i = 0; i < rows*cols; i++)
    {
        (*kernel)[i] = (*kernel)[i]/normalization;
    }
    
    
    
}


static void get_gaussian_kernel_3d(float** kernel, int rows, int cols, int depth, float sigma)
{
    /*

        kernel is given by the gaussian distribution:

        y = exp(||xi-xj||^2 /sigma^2)

    */

   //kernel allocation
    *kernel = (float*)malloc(sizeof(float)*rows*cols*depth);

    if (! *kernel)
    {
        return;
    }

    int x;
    int y;
    int z;

    int center_row = rows / 2;
    int center_col = cols / 2;
    int center_depth = depth / 2;

    float distance = 0;
    float normalization = 0;

    // Generate the kernel values.
    for (int k = 0; k < depth; k++)
    {
    
        for (int i = 0; i < rows; i++)
        {

            for (int j = 0; j < cols; j++)
            {

                x = i - center_row;
                y = j - center_col;
                z = k - center_depth;

                distance = x * x + y * y + z * z;

                (*kernel)[k*rows*cols + i * cols + j] = exp(-distance / (2 * sigma * sigma + 1E-16))*1E2;
                normalization += (*kernel)[k*rows*cols + i * cols + j];

                //std::cout<<(*kernel)[i*cols+j]<<" ";

            }

            //std::cout<<"\n";


        }
    
    }


    for (int i = 0; i < rows*cols*depth; i++)
    {
        (*kernel)[i] = (*kernel)[i]/normalization;
    }
      
}
#endif // KERNELS_H