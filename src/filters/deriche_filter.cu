#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/deriche_filter.h"

//based on: https://www.mathworks.com/matlabcentral/answers/7729-deriche-edge-detector


void get_deriche_vertical_kernel_2d(float** kernel, float alpha, int rows, int cols) {
    float c = ((1 - exp(-alpha)) * (1 - exp(-alpha))) / exp(-alpha);
    float k = (((1 - exp(-alpha)) * (1 - exp(-alpha))) * (alpha * alpha)) / (1 - 2 * alpha * exp(-alpha) - exp(-2 * alpha));

    *kernel = (float*)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            int index = i * cols + j;

            float x = j - (cols / 2);
            float y = i - (rows / 2);

            float X = (-c) * (x) * exp(-(alpha) * (fabs(x) + fabs(y))) * k * (alpha * (fabs(y) + 1));
            (*kernel)[index] = X / (alpha * alpha);
        }
    }

}


void get_deriche_horizontal_kernel_2d(float** kernel, float alpha, int rows, int cols) {
    float c = ((1 - exp(-alpha)) * (1 - exp(-alpha))) / exp(-alpha);
    float k = (((1 - exp(-alpha)) * (1 - exp(-alpha))) * (alpha * alpha)) / (1 - 2 * alpha * exp(-alpha) - exp(-2 * alpha));

    *kernel = (float*)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            int index = i * cols + j;

            float x = j - (cols / 2);
            float y = i - (rows / 2);

            float Y = (-c) * (y) * exp(-(alpha) * (fabs(x) + fabs(y))) * k * (alpha * (fabs(x) + 1));
            (*kernel)[index] = Y / (alpha * alpha);
        }
    }

}


void get_deriche_vertical_kernel_3d(float** kernel, float alpha, float beta, int rows, int cols, int depth) {
    float c = ((1 - exp(-alpha)) * (1 - exp(-alpha))) / exp(-alpha);
    float k = (((1 - exp(-alpha)) * (1 - exp(-alpha))) * (alpha * alpha)) / (1 - 2 * alpha * exp(-alpha) - exp(-2 * alpha));

    *kernel = (float*)malloc(rows * cols * depth * sizeof(float));

    for (int k = 0; k < depth; k++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int index = k * rows * cols + i * cols + j;

                float x = j - (cols / 2);
                float y = i - (rows / 2);
                float z = k - (depth / 2);

                float X = (-c) * (x) * exp(-(alpha) * (fabs(x) + fabs(y) + fabs(z))) * k * (alpha * (fabs(y) + fabs(z) + 1));
                (*kernel)[index] = X / (alpha * alpha);

            }
        }
    }

}


__global__ void deriche_gradient_magnitude_direction_2d(float* image, float* magnitude, uint8_t* direction,
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


//done
__global__ void deriche_non_maximum_supression_2d(float* magnitude, uint8_t* direction, int rows, int cols, int idz)
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


__global__ void deriche_thresholding_2d(float* image, float low, float high, int rows, int cols, int idz)
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
__global__ void deriche_hysteresis_2d(float* image, int rows, int cols, int idz)
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
void deriche_filtering(dtype* image, float* output,
                     int rows, int cols , int depth,
                     int rows_kernel, int cols_kernel, 
                     float alpha, float low_threshold, float high_threshold)
{

    /*

        gaussian step.

    */

    //device allocation for input and output images for the gaussian step.
    dtype* dev_image;
    cudaMalloc((void**)&dev_image, rows * cols * depth * sizeof(dtype));
    cudaMemcpy(dev_image, image, rows * cols * depth * sizeof(dtype), cudaMemcpyHostToDevice);

    //cuda kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + blockSize.y - 1) / blockSize.y, (cols + blockSize.x - 1) / blockSize.x);


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
    get_deriche_horizontal_kernel_2d(&horizontal_kernel,alpha,rows_kernel,cols_kernel);

    float* vertical_kernel;
    get_deriche_vertical_kernel_2d(&vertical_kernel, alpha, rows_kernel, cols_kernel);

    //allocate gradient kernels in device
    float* dev_horizontal_kernel;
    cudaMalloc((void**)&dev_horizontal_kernel, rows_kernel * cols_kernel * sizeof(float));
    cudaMemcpy(dev_horizontal_kernel, horizontal_kernel, rows_kernel * cols_kernel * sizeof(float), cudaMemcpyHostToDevice);

    float* dev_vertical_kernel;
    cudaMalloc((void**)&dev_vertical_kernel, rows_kernel * cols_kernel * sizeof(float));
    cudaMemcpy(dev_vertical_kernel, vertical_kernel, rows_kernel * cols_kernel * sizeof(float), cudaMemcpyHostToDevice);

    //Free memory allocated for host gradient kernels.
    free(horizontal_kernel);
    free(vertical_kernel);

    for (int k = 0; k < depth; k++)
    {
        deriche_gradient_magnitude_direction_2d<<<gridSize, blockSize>>>(dev_image, dev_magnitude, dev_direction,
                                                                 dev_horizontal_kernel, dev_vertical_kernel,
                                                                 rows, cols, k);
    }
    
    cudaDeviceSynchronize();

    cudaFree(dev_horizontal_kernel);
    cudaFree(dev_vertical_kernel);

    /*
    
        non-maximum supression step.
    
    */

   for (int k = 0; k < depth; k++)
   {
        deriche_non_maximum_supression_2d<<<gridSize,blockSize>>>(dev_magnitude, dev_direction, rows, cols, k);
   }

   cudaDeviceSynchronize();

   cudaFree(dev_direction);
   
   /*
   
        thresholding step.
   
   */

    for (int k = 0; k < depth; k++)
    {

        deriche_thresholding_2d<<<gridSize, blockSize>>>(dev_magnitude,low_threshold, high_threshold, rows, cols, k);
        
    }
    
    cudaDeviceSynchronize();


   /*
   
        hysteresis step.
   
   */

    for (int k = 0; k < depth; k++)
    {
        deriche_hysteresis_2d<<<gridSize, blockSize>>>(dev_magnitude, rows, cols, k);
    }

    cudaDeviceSynchronize();


    cudaMemcpy(output, dev_magnitude, rows * cols * depth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_magnitude);

}

template void deriche_filtering<float>(float* image, float* output,
                     int rows, int cols , int depth,
                     int rows_kernel, int cols_kernel, 
                     float alpha, float low_threshold, float high_threshold);


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
                    image[k * rows * cols + i * cols + j] = i/(j+1);
                }

                if (i==j)
                {
                    image[k * rows * cols + i * cols + j] = 0;
                }
                
        
                output[k * rows * cols + i * cols + j] = 0;
            }
        }

    }

    float alpha = 1.;
    float high = 5.;
    float low = 0.;

    int rows_kernel = 3;
    int cols_kernel = 3;

    deriche_filtering(image, output, rows, cols, slices, rows_kernel, cols_kernel, alpha, low, high);


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