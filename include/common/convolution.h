#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include<iostream>
#include<cuda_runtime.h>

/*

    2d general convolution.

*/

template<typename dtype>
__device__ void convolution2d(dtype* input,
                                float* output,
                                float* kernel,
                                int i, int j,
                                int xsize, int ysize,
                                int kx, int ky)
{
    float accumulation = 0;

    int inputX;
    int inputY;

    for(int m = 0; m < kx; m++)
    {
        for (int n = 0; n < ky; n++)
        {
            inputX = i - kx / 2 + m;
            inputY = j - ky / 2 + n;

            if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize)
            {
                accumulation += kernel[m * ky + n] * input[inputX * ysize + inputY];
            }

            else
            {   
                // Reflect padding
                if (inputX < 0)
                    inputX = -inputX;
                else if (inputX >= xsize)
                    inputX = 2 * xsize - inputX - 1;

                if (inputY < 0)
                    inputY = -inputY;
                else if (inputY >= ysize)
                    inputY = 2 * ysize - inputY - 1;

                accumulation += kernel[m * ky + n] * input[inputX * ysize + inputY];
            }
            
        }
    }

    *output = (float) accumulation;
}

/*

    3d general convolution.

*/
template<typename dtype>
__device__ void convolution3d(dtype* input,
                            float* output,
                            float* kernel,
                            int i, int j, int k,
                            int xsize, int ysize, int zsize,
                            int kx, int ky, int kz)
{
    float accumulation = 0;

    int inputY;
    int inputX;
    int inputZ;
    
    for (int l = 0; l < kz; l++)
    {

        for(int m = 0; m < kx; m++)
        {
            
                for (int n = 0; n < ky; n++)
            {
                //this is needed to compute everything with respect to the center of the kernel.
                inputX = i - kx / 2 + m;
                inputY = j - ky / 2 + n;
                inputZ = k - kz / 2 + l;

                //checks for boundaries.
                if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize && inputZ >= 0 && inputZ < zsize)
                {
                    accumulation += kernel[(l * kx * ky) + (m * ky) + n] * input[(inputZ * xsize * ysize) + (inputX * ysize) + inputY];
                }
            
                //make a padding function to substitute this line of code.
                else
                {
                    // Reflect padding
                    if (inputX < 0)
                    {
                        inputX = -inputX;
                    }

                    else if (inputX >= xsize)
                    {
                        inputX = 2 * xsize - inputX - 1;
                    }


                    if (inputY < 0)
                    {
                        inputY = -inputY;
                    }

                    else if (inputY >= ysize)
                    {
                        inputY = 2 * ysize - inputY - 1;
                    }

                    if (inputZ < 0)
                    {
                        inputZ = - inputZ;
                    }

                    else if (inputZ>=zsize)
                    {
                        inputZ = 2 * zsize - inputZ -1;
                    }
                    

                    accumulation += kernel[(l * kx * ky) + (m * ky) + n] * input[(inputZ * xsize * ysize) + (inputX * ysize) + inputY];

                }

            }

        }

    }

    *output = (float) accumulation;

}


#endif // CONVOLUTION_H
