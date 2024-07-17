#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/canny_filter.h"

//based on: https://github.com/arashsm79/parallel-canny-edge-detector/tree/main


void get_horizontal_kernel_2d(float** kernel)
{
    /*

        Horizontal kernel has the form:

                   1  0 -1
                   2  0 -2
                   1  0 -1

    */

    *kernel = (float*)malloc(sizeof(float)*9);

    if (! *kernel)
    {
        return;
    }
    

    (*kernel)[0] = 1;
    (*kernel)[1] = 0;
    (*kernel)[2] = -1;

    (*kernel)[3] = 2;
    (*kernel)[4] = 0;
    (*kernel)[5] = -2;

    (*kernel)[6] = 1;
    (*kernel)[7] = 0;
    (*kernel)[8] = -1;
    

}


void get_vertical_kernel_2d(float** kernel)
{
    /*

        Vertical kernel has the form:

                   1  2  1
                   0  0  0
                  -1 -2 -1

    */

    *kernel = (float*)malloc(sizeof(float)*9);

    if (! *kernel)
    {
        return;
    }

    (*kernel)[0] = 1;
    (*kernel)[1] = 2;
    (*kernel)[2] = 1;

    (*kernel)[3] = 0;
    (*kernel)[4] = 0;
    (*kernel)[5] = 0;

    (*kernel)[6] = -1;
    (*kernel)[7] = -2;
    (*kernel)[8] = -1;

}

__global__ void gradient_magnitude_direction_2d(float* image, float* magnitude, uint8_t* direction,
                                                float* horizontal_kernel, float* vertical_kernel,
                                                int rows, int cols, int idz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        int index = idz * rows * cols + idx * cols + idy;

        uint8_t temp = 0;
        float grad_x = 0;
        float grad_y = 0;

        convolution_2d(image + idz * rows * cols, &grad_x, horizontal_kernel, idx, idy, rows, cols, 3, 3);
        convolution_2d(image + idz * rows * cols, &grad_y, vertical_kernel, idx, idy, rows, cols, 3, 3);

        if (grad_x == 0 || grad_y == 0)
        {
            magnitude[index] = (float)0.;
        }

        else
        {
            magnitude[index] = (float)sqrtf(grad_x*grad_x + grad_y*grad_y);

            float theta = atan2f(grad_y,grad_x) * (360.0f/PI);
            
            if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
            {
                temp = 1;
            }
            
            else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
            {
                temp = 2;
            }

            else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
            {
                temp = 3;
            }

            else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
            {
                temp = 4;
            }
            
            
        }

        direction[index] = temp;
        
        

    }


}

//can be optimized -- > improve borders by reflect -- > already done it in the gradient step, no need here
//therefore its done
__global__ void non_maximum_supression_2d(float* magnitude, uint8_t* direction, int rows, int cols, int idz)
{

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        int index = idz * rows * cols + idx * cols + idy;
        
        switch (direction[index])
        {
            case 1:
                
                if (magnitude[index - 1] >= magnitude[index] ||
                    magnitude[index + 1] > magnitude[index])
                {
                    magnitude[index] = 0;
                }
                
                break;
            
            case 2:

                if (magnitude[index - (cols-1)] >= magnitude[index] ||
                    magnitude[index + (cols-1)] > magnitude[index])
                {
                    magnitude[index] = 0;
                }

                break;

            case 3:

                if (magnitude[index - cols] >= magnitude[index] ||
                    magnitude[index + cols] > magnitude[index])
                {
                    magnitude[index] = 0;
                }

                break;


            case 4:

                if (magnitude[index - (cols+1)] >= magnitude[index] ||
                    magnitude[index + (cols+1)] > magnitude[index])
                {
                    magnitude[index] = 0;
                }

                break;

            default:

                magnitude[index] = 0;
                
                break;
            
                
        }



    }
    

}


__global__ void thresholding_2d(float* image, float low, float high, int rows, int cols, int idz)
{

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy <cols)
    {
        int index = idz * rows * cols + idx * cols + idy;

        //strong edge.
        if (image[index] > high)
        {
            image[index] = 255;
        }

        //weak edge.
        else if (image[index] > low)
        {
            image[index] = 100;
        }

        //not an edge.
        else
        {
            image[index] = 0;
        }
        
        
        
    }
    


}

//done
__global__ void hysteresis_2d(float* image, int rows, int cols, int idz)
{

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < rows && idy < cols)
    {
        int index = idz * rows * cols + idx * cols + idy;

        if (image[index] == 100)//uma forma melhor seria usando a soma dos valores e utilizar o resto da divisão, para analisar se a condição é valida==> muito mais rapido que ifs
        {
            if (image[index - 1] == 255 || image[index + 1] == 255 ||
                image[index - cols] == 255 || image[index + cols] ||
                image[index - cols - 1] == 255 || image[index - cols + 1] == 255 || 
                image[index + cols - 1] == 255 || image[index + cols + 1] == 255)
            {
                image[index] = 255;
            }

            else
            {
                image[index] = 0;
            }
            
            
        }
        
    }
    


}



template<typename dtype>
void canny_filtering(dtype* image, float* output,
                     int rows, int cols , int depth,
                     float sigma, float low_threshold, float high_threshold)
{

    /*

        gaussian step.

    */

    //device allocation for input and output images for the gaussian step.
    dtype* dev_image;
    float* dev_output;
    cudaMalloc((void**)&dev_image, rows * cols * depth * sizeof(dtype));
    cudaMalloc((void**)&dev_output, rows * cols * depth * sizeof(float));
    cudaMemcpy(dev_image, image, rows * cols * depth * sizeof(dtype), cudaMemcpyHostToDevice);


    // get gaussian kernel size
    int rows_gaussian_kernel = (int)ceil(2*sigma+1);
    int cols_gaussian_kernel = rows_gaussian_kernel;

    //get gaussian kernel.
    float* gaussian_kernel;
    get_gaussian_kernel_2d(&gaussian_kernel, rows_gaussian_kernel, cols_gaussian_kernel, sigma);

    //device allocation for the gaussian kernel
    float* dev_gaussian_kernel;
    cudaMalloc((void**)&dev_gaussian_kernel, rows_gaussian_kernel * cols_gaussian_kernel * sizeof(float));
    cudaMemcpy(dev_gaussian_kernel, gaussian_kernel, rows_gaussian_kernel * cols_gaussian_kernel * sizeof(float), cudaMemcpyHostToDevice);

    //Free host gaussian kernel
    free(gaussian_kernel);

    //cuda kernel configuration
    dim3 blockSize(32, 32);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x);

    //apply gaussian blur
    for (int k = 0; k < depth; k++)
    {
            gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_gaussian_kernel,
                                                               k, rows, cols, depth,
                                                               rows_gaussian_kernel, cols_gaussian_kernel);
    }

    //sync with host
    cudaDeviceSynchronize();

    //manage device memory.
    cudaFree(dev_image);
    cudaFree(dev_gaussian_kernel);

    /*
    
        gradient step.
    
    */

    //device allocation for the gradient magnitude and direction.
    float* dev_magnitude;
    uint8_t* dev_direction;
    cudaMalloc((void**)&dev_magnitude, rows * cols * depth * sizeof(float));
    cudaMalloc((void**)&dev_direction, rows * cols * depth * sizeof(uint8_t));
    
    //get gradient kernels.
    float* horizontal_kernel;
    get_horizontal_kernel_2d(&horizontal_kernel);

    float* vertical_kernel;
    get_vertical_kernel_2d(&vertical_kernel);

    //allocate gradient kernels in device
    float* dev_horizontal_kernel;
    cudaMalloc((void**)&dev_horizontal_kernel, 9 * sizeof(float));
    cudaMemcpy(dev_horizontal_kernel, horizontal_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    float* dev_vertical_kernel;
    cudaMalloc((void**)&dev_vertical_kernel, 9 * sizeof(float));
    cudaMemcpy(dev_vertical_kernel, vertical_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    //Free memory allocated for host gradient kernels.
    free(horizontal_kernel);
    free(vertical_kernel);

    for (int k = 0; k < depth; k++)
    {
        gradient_magnitude_direction_2d<<<gridSize, blockSize>>>(dev_output, dev_magnitude, dev_direction,
                                                                 dev_horizontal_kernel, dev_vertical_kernel,
                                                                 rows, cols, k);
    }
    
    cudaDeviceSynchronize();

    cudaFree(dev_output);
    cudaFree(dev_horizontal_kernel);
    cudaFree(dev_vertical_kernel);

    /*
    
        non-maximum supression step.
    
    */

   for (int k = 0; k < depth; k++)
   {
        non_maximum_supression_2d<<<gridSize,blockSize>>>(dev_magnitude, dev_direction, rows, cols, k);
   }

   cudaDeviceSynchronize();

   cudaFree(dev_direction);
   
   /*
   
        thresholding step.
   
   */

    for (int k = 0; k < depth; k++)
    {

        thresholding_2d<<<gridSize, blockSize>>>(dev_magnitude,low_threshold, high_threshold, rows, cols, k);
        
    }
    
    cudaDeviceSynchronize();


   /*
   
        hysteresis step.
   
   */

    for (int k = 0; k < depth; k++)
    {
        hysteresis_2d<<<gridSize, blockSize>>>(dev_magnitude, rows, cols, k);
    }

    cudaDeviceSynchronize();


    cudaMemcpy(output, dev_magnitude, rows * cols * depth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_magnitude);

}

// Explicit instantiation for float
template void canny_filtering<float>(float* image, float* output,
                                    int rows, int cols , int depth,
                                    float sigma, float low_threshold, float high_threshold);

/*

int main()
{
    int rows = 50;
    int cols = 50;
    int slices = 1;

    static float* image;
    image = (float*)malloc(slices*rows*cols*sizeof(int));

    static float* output;
    output = (float*)malloc(slices*rows*cols*sizeof(int));

    for (int k = 0; k < slices; k++)
    {

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i!=j)
                {
                    image[k * rows * cols + i * cols + j] = i+j;
                }

                if (i==j)
                {
                    image[k * rows * cols + i * cols + j] = 0;
                }
                
        
                output[k * rows * cols + i * cols + j] = 0;
            }
        }

    }

    float sigma = 1.;
    float high = 5.;
    float low = 0.;
    canny_filtering(image,output,rows,cols,slices,sigma,low, high);


    for (int k = 0; k < slices; k++)
    {

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout<<image[k*rows*cols + i*cols +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    
    std::cout<<"\n";
    
    for (int k = 0; k < slices; k++)
    {

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout<<output[k*rows*cols + i*cols +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    

    

    return 0;
}
*/