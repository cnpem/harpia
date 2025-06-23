#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/canny_filter.h"

//based on: https://github.com/arashsm79/parallel-canny-edge-detector/tree/main


void get_horizontal_kernel_2d(float** hostKernel)
{
    /*

        Horizontal hostKernel has the form:

                   1  0 -1
                   2  0 -2
                   1  0 -1

    */

    *hostKernel = (float*)malloc(sizeof(float)*9);

    if (! *hostKernel)
    {
        return;
    }
    

    (*hostKernel)[0] = 1;
    (*hostKernel)[1] = 0;
    (*hostKernel)[2] = -1;

    (*hostKernel)[3] = 2;
    (*hostKernel)[4] = 0;
    (*hostKernel)[5] = -2;

    (*hostKernel)[6] = 1;
    (*hostKernel)[7] = 0;
    (*hostKernel)[8] = -1;
    

}




void get_vertical_kernel_2d(float** hostKernel)
{
    /*

        Vertical hostKernel has the form:

                   1  2  1
                   0  0  0
                  -1 -2 -1

    */

    *hostKernel = (float*)malloc(sizeof(float)*9);

    if (! *hostKernel)
    {
        return;
    }

    (*hostKernel)[0] = -1;
    (*hostKernel)[1] = -2;
    (*hostKernel)[2] = -1;

    (*hostKernel)[3] = 0;
    (*hostKernel)[4] = 0;
    (*hostKernel)[5] = 0;

    (*hostKernel)[6] = 1;
    (*hostKernel)[7] = 2;
    (*hostKernel)[8] = 1;

}

__global__ void gradient_magnitude_direction_2d(float* deviceImage, float* deviceMagnitude, uint8_t* deviceGradientDirection,
                                                float* deviceKernelHorizontal, float* deviceKernelVertical,
                                                int xsize, int ysize, int idz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        int imageIndex = idz * xsize * ysize + idx * ysize + idy;

        uint8_t temp = 0;
        float gradientX = 0;
        float gradientY = 0;

        convolution2d(deviceImage + idz * xsize * ysize, &gradientX, deviceKernelHorizontal, idx, idy, xsize, ysize, 3, 3);
        convolution2d(deviceImage + idz * xsize * ysize, &gradientY, deviceKernelVertical, idx, idy, xsize, ysize, 3, 3);

        if (gradientX == 0 || gradientY == 0)
        {
            deviceMagnitude[imageIndex] = 0.0f;
        }
        else
        {
            deviceMagnitude[imageIndex] = sqrtf(gradientX * gradientX + gradientY * gradientY);

            // Calculate theta and determine the direction
            float theta = atan2f(gradientY , gradientX) * (180.0f / PI);
            
            // Normalize theta to be within [0, 180]
            if (theta < 0) theta += 180;

            // Assign direction based on theta
            if ((theta >= 0 && theta < 22.5) || (theta >= 157.5 && theta < 180))
            {
                temp = 1; // 0 degrees
            }
            else if (theta >= 22.5 && theta < 67.5)
            {
                temp = 2; // 45 degrees
            }
            else if (theta >= 67.5 && theta < 112.5)
            {
                temp = 3; // 90 degrees
            }
            else if (theta >= 112.5 && theta < 157.5)
            {
                temp = 4; // 135 degrees
            }
        }

        deviceGradientDirection[imageIndex] = temp;
    }
}

__global__ void non_maximum_supression_2d(float* deviceMagnitude, uint8_t* deviceGradientDirection, int xsize, int ysize, int idz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        int imageIndex = idz * xsize * ysize + idx * ysize + idy;
        
        float currentMag = deviceMagnitude[imageIndex];
        
        switch (deviceGradientDirection[imageIndex])
        {
            case 1: // 0 degrees
                if (idx > 0 && idx < xsize - 1) {
                    if (currentMag < deviceMagnitude[imageIndex - 1] || currentMag < deviceMagnitude[imageIndex + 1]) {
                        deviceMagnitude[imageIndex] = 0;
                    }
                }
                break;
            
            case 2: // 45 degrees
                if (idx > 0 && idx < xsize - 1 && idy > 0 && idy < ysize - 1) {
                    if (currentMag < deviceMagnitude[imageIndex - (ysize - 1)] || currentMag < deviceMagnitude[imageIndex + (ysize + 1)]) {
                        deviceMagnitude[imageIndex] = 0;
                    }
                }
                break;

            case 3: // 90 degrees
                if (idy > 0 && idy < ysize - 1) {
                    if (currentMag < deviceMagnitude[imageIndex - ysize] || currentMag < deviceMagnitude[imageIndex + ysize]) {
                        deviceMagnitude[imageIndex] = 0;
                    }
                }
                break;

            case 4: // 135 degrees
                if (idx > 0 && idx < xsize - 1 && idy > 0 && idy < ysize - 1) {
                    if (currentMag < deviceMagnitude[imageIndex - (ysize + 1)] || currentMag < deviceMagnitude[imageIndex + (ysize - 1)]) {
                        deviceMagnitude[imageIndex] = 0;
                    }
                }
                break;

            default:
                deviceMagnitude[imageIndex] = 0;
                break;
        }
    }
}


__global__ void thresholding_2d(float* deviceImage, float low, float high, int xsize, int ysize, int idz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        int imageIndex = idz * xsize * ysize + idx * ysize + idy;  // Corrected indexing for row-major order

        float pixelValue = deviceImage[imageIndex];

        // Strong edge
        if (pixelValue >= high)
        {
            deviceImage[imageIndex] = 255; // Strong edge value
        }
        // Weak edge
        else if (pixelValue >= low)
        {
            deviceImage[imageIndex] = 100; // Weak edge value
        }
        // Not an edge
        else
        {
            deviceImage[imageIndex] = 0; // Non-edge value
        }
    }
}


//done
__global__ void hysteresis_2d(float* deviceImage, int xsize, int ysize, int idz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        int imageIndex = idz * xsize * ysize + idx * ysize + idy;

        // Only process weak edge pixels
        if (deviceImage[imageIndex] == 100)
        {
            bool isConnectedToStrongEdge = false;

            // Single boundary check to handle all neighbors
            if ((idx > 0 && idx < xsize - 1) && (idy > 0 && idy < ysize - 1))
            {
                // Check the 8 neighbors for strong edges (255)
                if (deviceImage[imageIndex - 1] == 255 ||                        // Left
                    deviceImage[imageIndex + 1] == 255 ||                        // Right
                    deviceImage[imageIndex - ysize] == 255 ||                   // Top
                    deviceImage[imageIndex + ysize] == 255 ||                   // Bottom
                    deviceImage[imageIndex - ysize - 1] == 255 ||               // Top-left
                    deviceImage[imageIndex - ysize + 1] == 255 ||               // Top-right
                    deviceImage[imageIndex + ysize - 1] == 255 ||               // Bottom-left
                    deviceImage[imageIndex + ysize + 1] == 255)                 // Bottom-right
                {
                    isConnectedToStrongEdge = true;
                }
            }
            else
            {
                // Handle edge cases (pixels along the borders) separately
                if (idx > 0 && deviceImage[imageIndex - 1] == 255) isConnectedToStrongEdge = true;                 // Left
                if (idx < xsize - 1 && deviceImage[imageIndex + 1] == 255) isConnectedToStrongEdge = true;         // Right
                if (idy > 0 && deviceImage[imageIndex - ysize] == 255) isConnectedToStrongEdge = true;             // Top
                if (idy < ysize - 1 && deviceImage[imageIndex + ysize] == 255) isConnectedToStrongEdge = true;     // Bottom
                if (idx > 0 && idy > 0 && deviceImage[imageIndex - ysize - 1] == 255) isConnectedToStrongEdge = true;           // Top-left
                if (idx < xsize - 1 && idy > 0 && deviceImage[imageIndex - ysize + 1] == 255) isConnectedToStrongEdge = true;   // Top-right
                if (idx > 0 && idy < ysize - 1 && deviceImage[imageIndex + ysize - 1] == 255) isConnectedToStrongEdge = true;   // Bottom-left
                if (idx < xsize - 1 && idy < ysize - 1 && deviceImage[imageIndex + ysize + 1] == 255) isConnectedToStrongEdge = true; // Bottom-right
            }

            // Update the weak edge pixel based on connectivity to strong edges
            if (isConnectedToStrongEdge)
            {
                deviceImage[imageIndex] = 255; // Mark as strong edge
            }
            else
            {
                deviceImage[imageIndex] = 0; // Mark as non-edge
            }
        }
    }
}


template<typename dtype>
void canny_filtering(dtype* hostImage, float* hostOutput,
                     int xsize, int ysize , int zsize,
                     float sigma, float lowThreshold, float highThreshold)
{

    /*

        gaussian step.

    */

    //device allocation for input and hostOutput images for the gaussian step.
    dtype* deviceImage;
    float* deviceOutput;
    cudaMalloc((void**)&deviceImage, xsize * ysize * zsize * sizeof(dtype));
    cudaMalloc((void**)&deviceOutput, xsize * ysize * zsize * sizeof(float));
    cudaMemcpy(deviceImage, hostImage, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);


    // get gaussian kernel size
    int nx = (int)ceil(4*sigma+0.5);
    int ny = nx;

    //get gaussian kernel.
    double* gaussian_kernel;
    get_gaussian_kernel_2d(&gaussian_kernel, nx, ny, sigma);

    //device allocation for the gaussian kernel
    double* deviceKernel;
    cudaMalloc((void**)&deviceKernel, nx * ny * sizeof(double));
    cudaMemcpy(deviceKernel, gaussian_kernel, nx * ny * sizeof(double), cudaMemcpyHostToDevice);

    //Free host gaussian kernel
    free(gaussian_kernel);

    //cuda kernel configuration
    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //apply gaussian blur
    for (int k = 0; k < zsize; k++)
    {
            gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, deviceKernel,
                                                               k, xsize, ysize, zsize,
                                                               nx, ny);
    }

    //sync with host
    cudaDeviceSynchronize();

    //manage device memory.
    cudaFree(deviceImage);
    cudaFree(deviceKernel);

    /*
    
        gradient step.
    
    */

    //device allocation for the gradient deviceMagnitude and deviceGradientDirection.
    float* deviceMagnitude;
    uint8_t* deviceGradientDirection;
    cudaMalloc((void**)&deviceMagnitude, xsize * ysize * zsize * sizeof(float));
    cudaMalloc((void**)&deviceGradientDirection, xsize * ysize * zsize * sizeof(uint8_t));
    
    //get gradient kernels.
    float* hostKernelHorizontal;
    get_horizontal_kernel_2d(&hostKernelHorizontal);

    float* hostKernelVertical;
    get_vertical_kernel_2d(&hostKernelVertical);

    //allocate gradient kernels in device
    float* deviceKernelHorizontal;
    cudaMalloc((void**)&deviceKernelHorizontal, 9 * sizeof(float));
    cudaMemcpy(deviceKernelHorizontal, hostKernelHorizontal, 9 * sizeof(float), cudaMemcpyHostToDevice);

    float* deviceKernelVertical;
    cudaMalloc((void**)&deviceKernelVertical, 9 * sizeof(float));
    cudaMemcpy(deviceKernelVertical, hostKernelVertical, 9 * sizeof(float), cudaMemcpyHostToDevice);

    //Free memory allocated for host gradient kernels.
    free(hostKernelHorizontal);
    free(hostKernelVertical);

    for (int k = 0; k < zsize; k++)
    {
        gradient_magnitude_direction_2d<<<gridSize, blockSize>>>(deviceOutput, deviceMagnitude, deviceGradientDirection,
                                                                 deviceKernelHorizontal, deviceKernelVertical,
                                                                 xsize, ysize, k);
    }
    
    cudaDeviceSynchronize();

    cudaFree(deviceOutput);
    cudaFree(deviceKernelHorizontal);
    cudaFree(deviceKernelVertical);

    /*
    
        non-maximum supression step.
    
    */

   for (int k = 0; k < zsize; k++)
   {
        non_maximum_supression_2d<<<gridSize,blockSize>>>(deviceMagnitude, deviceGradientDirection, xsize, ysize, k);
   }

   cudaDeviceSynchronize();

   cudaFree(deviceGradientDirection);
   
   /*
   
        thresholding step.
   
   */

    for (int k = 0; k < zsize; k++)
    {

        thresholding_2d<<<gridSize, blockSize>>>(deviceMagnitude,lowThreshold, highThreshold, xsize, ysize, k);
        
    }
    
    cudaDeviceSynchronize();


   /*
   
        hysteresis step.
   
   */

    for (int k = 0; k < zsize; k++)
    {
        hysteresis_2d<<<gridSize, blockSize>>>(deviceMagnitude, xsize, ysize, k);
    }

    cudaDeviceSynchronize();


    cudaMemcpy(hostOutput, deviceMagnitude, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceMagnitude);

}

// Explicit instantiation
template void canny_filtering<float>(float* hostImage, float* hostOutput,
                                    int xsize, int ysize , int zsize,
                                    float sigma, float lowThreshold, float highThreshold);

template void canny_filtering<int>(int* hostImage, float* hostOutput,
                                    int xsize, int ysize , int zsize,
                                    float sigma, float lowThreshold, float highThreshold);

template void canny_filtering<unsigned int>(unsigned int* hostImage, float* hostOutput,
                                    int xsize, int ysize , int zsize,
                                    float sigma, float lowThreshold, float highThreshold);

/*

int main()
{
    int xsize = 50;
    int ysize = 50;
    int slices = 1;

    static float* hostImage;
    hostImage = (float*)malloc(slices*xsize*ysize*sizeof(int));

    static float* hostOutput;
    hostOutput = (float*)malloc(slices*xsize*ysize*sizeof(int));

    for (int k = 0; k < slices; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                if (i!=j)
                {
                    hostImage[k * xsize * ysize + i * ysize + j] = i+j;
                }

                if (i==j)
                {
                    hostImage[k * xsize * ysize + i * ysize + j] = 0;
                }
                
        
                hostOutput[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }

    float sigma = 1.;
    float high = 5.;
    float low = 0.;
    canny_filtering(hostImage,hostOutput,xsize,ysize,slices,sigma,low, high);


    for (int k = 0; k < slices; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                std::cout<<hostImage[k*xsize*ysize + i*ysize +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    
    std::cout<<"\n";
    
    for (int k = 0; k < slices; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                std::cout<<hostOutput[k*xsize*ysize + i*ysize +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    

    

    return 0;
}
*/