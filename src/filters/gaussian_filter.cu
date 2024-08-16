#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/gaussian_filter.h"


template<typename dtype>
__global__ void gaussian_filter_kernel_2d(dtype* image, float* output,float* dev_kernel,
                                          int idz, int xsize, int ysize, int zsize, int kx, int ky)
{   

    //threads indices
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // general matrix convolution for each pixel of the image.
    if (idx<xsize && idy<ysize)
    {   
        //temp variable
        float temp;

        //convolution.
        convolution2d(image + idz * xsize * ysize,&temp,dev_kernel,idx,idy,xsize,ysize,kx,ky); 

        output[idz * xsize * ysize + idx * ysize + idy] = (float)temp;
        
    }
      
}


template<typename dtype>
__global__ void gaussian_filter_kernel_3d(dtype* image, float* output,float* dev_kernel,
                                          int xsize, int ysize, int zsize,int kx, int ky, int kz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < xsize && idy < ysize && idz < zsize)
    {
        float temp;

        convolution3d(image, &temp, dev_kernel, idx, idy, idz, xsize, ysize, zsize, kx, ky, kz);

        output[idz * xsize * ysize + idx * ysize + idy] = (float)temp;
    }
}


template __global__ void gaussian_filter_kernel_2d<int>(int* image, float* output, float* dev_kernel,int idz, int xsize, int ysize, int zsize, int kx, int ky);
template __global__ void gaussian_filter_kernel_2d<float>(float* image, float* output, float* dev_kernel, int idz, int xsize, int ysize, int zsize, int kx, int ky);

template __global__ void gaussian_filter_kernel_3d<int>(int* image, float* output,float* dev_kernel,int xsize, int ysize, int zsize,int kx, int ky, int kz);
template __global__ void gaussian_filter_kernel_3d<float>(float* image, float* output,float* dev_kernel,int xsize, int ysize, int zsize,int kx, int ky, int kz);


template<typename dtype>
void gaussian_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, float sigma, bool type)
{

    dtype* dev_image;
    float* dev_output;
    cudaMalloc((void**)&dev_image, xsize * ysize * zsize * sizeof(dtype));
    cudaMalloc((void**)&dev_output, xsize * ysize * zsize * sizeof(float));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);
    

    if (type == false)
    {
        //kernel size
        int kx = (int)ceil(2*sigma+1);
        int ky = kx;

        float* kernel;
        get_gaussian_kernel_2d(&kernel,kx,ky,sigma);
        

        float* dev_kernel;
        cudaMalloc((void**)&dev_kernel, kx * ky * sizeof(float));
        cudaMemcpy(dev_kernel, kernel, kx * ky * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(32, 32);
        dim3 gridSize( (xsize + blockSize.y - 1) / blockSize.y,(ysize + blockSize.x - 1) / blockSize.x);

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < zsize; ++k)
        {
            gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel, k, xsize, ysize, zsize, kx, ky);

            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

        cudaFree(dev_kernel);   
    }

    else
    {
        //kernel size
        int kx = (int)ceil(2*sigma+1);
        int ky = kx;
        int kz = kx;

        float* kernel;
        get_gaussian_kernel_3d(&kernel,kx,ky, kz, sigma);
        

        float* dev_kernel;
        cudaMalloc((void**)&dev_kernel, kx * ky * kz * sizeof(float));
        cudaMemcpy(dev_kernel, kernel, kx * ky * kz * sizeof(float), cudaMemcpyHostToDevice);


        dim3 blockSize(8, 8, 8);
        dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y,(ysize + blockSize.x - 1) / blockSize.x, (zsize + blockSize.z - 1) / blockSize.z);

        auto start = std::chrono::high_resolution_clock::now();

        gaussian_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output,dev_kernel,xsize, ysize, zsize, kx, ky, kz);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

        cudaFree(dev_kernel);
    }

    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
}


// Explicit instantiation
template void gaussian_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize, float sigma, bool type);
template void gaussian_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize, float sigma, bool type);
template void gaussian_filtering<unsigned int>(unsigned int* image, float* output, int xsize, int ysize, int zsize, float sigma, bool type);

/*
int main()
{
    int xsize = 512;
    int ysize = 512;
    int zsize = 512;

    static float* image;
    image = (float*)malloc(zsize*xsize*ysize*sizeof(int));

    static float* output;
    output = (float*)malloc(zsize*xsize*ysize*sizeof(int));

    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                if (i!=j)
                {
                    image[k * xsize * ysize + i * ysize + j] = 1;
                }

                if (i==j)
                {
                    image[k * xsize * ysize + i * ysize + j] = i+j;
                }
                
        
                output[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }

    float sigma = 6.;
    gaussian_filtering(image,output,xsize,ysize,zsize,sigma,true);
    
    
    for (int k = 0; k < zsize; k++)
    {

        for (int i = 0; i < xsize; i++)
        {
            for (int j = 0; j < ysize; j++)
            {
                std::cout<<output[k*xsize*ysize + i*ysize +j]<<" ";
            }

            std::cout<<"\n";
        }

        std::cout<<"\n";

    }
    

    

    return 0;
}
*/