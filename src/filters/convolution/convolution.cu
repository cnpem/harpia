#include<iostream>
#include<cuda_runtime.h>
#include"convolution.h"

/*

    2d general convolution.

*/

template<typename dtype>
__device__ void convolution_2d(dtype* input,
                                float* output,
                                float* kernel,
                                int i, int j,
                                int rows, int cols,
                                int rows_kernel, int cols_kernel)
{
    float accumulation = 0;

    int input_row;
    int input_col;

    for(int m = 0; m < rows_kernel; m++)
    {
        for (int n = 0; n < cols_kernel; n++)
        {
            input_row = i - rows_kernel / 2 + m;
            input_col = j - cols_kernel / 2 + n;

            if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols)
            {
                accumulation += kernel[m * cols_kernel + n] * input[input_row * cols + input_col];
            }

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

                accumulation += kernel[m * cols_kernel + n] * input[input_row * cols + input_col];
            }
            
        }
    }

    *output = (float) accumulation;
}

/*

    3d general convolution.

*/
template<typename dtype>
__device__ void convolution_3d(dtype* input,
                            float* output,
                            float* kernel,
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

                //checks for boundaries.
                if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols && input_depth >= 0 && input_depth < depth)
                {
                    accumulation += kernel[(l * rows_kernel * cols_kernel) + (m * cols_kernel) + n] * input[(input_depth * rows * cols) + (input_row * cols) + input_col];
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
                    

                    accumulation += kernel[(l * rows_kernel * cols_kernel) + (m * cols_kernel) + n] * input[(input_depth * rows * cols) + (input_row * cols) + input_col];

                }

            }

        }

    }

    *output = (float) accumulation;

}



