#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/filters/sobel_filter.h"


void get_sobel_horizontal_kernel_2d(float** kernel)
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


void get_sobel_vertical_kernel_2d(float** kernel)
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


void get_sobel_horizontal_kernel_3d(float** kernel)
{
    //adapted from: https://www.hindawi.com/journals/mpe/2016/4904279/

    /*
            
                 +---------------+
                /    -2  0  2   /|
               /     -3  0  3  / |
              /      -2  0  2 /  |
             +---------------+   |
            /    -3  0  3   /|  /
           /     -6  0  6  / | /
          /      -3  0  3 /  |/ 
         +---------------+   +
         |    -2  0  2   |  /
         |    -3  0  3   | /
         |    -2  0  2   |/
         +---------------+

    
    */



    //kernel allocation.
    *kernel = (float*)malloc(27*sizeof(float));

    if (! *kernel)
    {
        return;
    }

    //first plane
    (*kernel)[0] = -2;
    (*kernel)[1] =  0;
    (*kernel)[2] =  2;

    (*kernel)[3] = -3;
    (*kernel)[4] =  0;
    (*kernel)[5] =  3;

    (*kernel)[6] = -2;
    (*kernel)[7] =  0;
    (*kernel)[8] =  2;

    //second plane
    (*kernel)[9] = -3;
    (*kernel)[10] =  0;
    (*kernel)[11] = -3;

    (*kernel)[12] = -6;
    (*kernel)[13] =  0;
    (*kernel)[14] =  6;

    (*kernel)[15] = -3;
    (*kernel)[16] =  0;
    (*kernel)[17] =  3;

    //third plane
    (*kernel)[18] = -2;
    (*kernel)[19] =  0;
    (*kernel)[20] =  2;

    (*kernel)[21] = -3;
    (*kernel)[22] =  0;
    (*kernel)[23] =  3;

    (*kernel)[24] = -2;
    (*kernel)[25] =  0;
    (*kernel)[26] =  2;
    

}


void get_sobel_vertical_kernel_3d(float** kernel)
{
    /*
            
                 +---------------+
                /    -2 -3 -2   /|
               /      0  0  0  / |
              /       2  3  2 /  |
             +---------------+   |
            /  -3  -6  -3   /|  /
           /    0   0  0  /  | /
          /     3   6  3 /   |/ 
         +---------------+   +
         |  -2  -3  -2   |  /
         |   0   0   0   | /
         |   2   3   2   |/
         +---------------+

    
    */

        //kernel allocation.
    *kernel = (float*)malloc(27*sizeof(float));

    if (! *kernel)
    {
        return;
    }

    //first plane
    (*kernel)[0] = -2;
    (*kernel)[1] = -3;
    (*kernel)[2] = -2;

    (*kernel)[3] =  0;
    (*kernel)[4] =  0;
    (*kernel)[5] =  0;

    (*kernel)[6] =  2;
    (*kernel)[7] =  3;
    (*kernel)[8] =  2;

    //second plane
    (*kernel)[9] = -3;
    (*kernel)[10] = -6;
    (*kernel)[11] = -3;

    (*kernel)[12] =  0;
    (*kernel)[13] =  0;
    (*kernel)[14] =  0;

    (*kernel)[15] =  3;
    (*kernel)[16] =  6;
    (*kernel)[17] =  3;

    //third plane
    (*kernel)[18] = -2;
    (*kernel)[19] = -3;
    (*kernel)[20] = -2;

    (*kernel)[21] =  0;
    (*kernel)[22] =  0;
    (*kernel)[23] =  0;

    (*kernel)[24] =  2;
    (*kernel)[25] =  3;
    (*kernel)[26] =  2;
}


void get_sobel_depth_kernel_3d(float** kernel)
{

      /*
            
                 +---------------+
                /    2  3  2    /|
               /     3  6  3   / |
              /      2  3  2  /  |
             +---------------+   |
            /     0  0  0   /|  /
           /      0  0  0  / | /
          /       0  0  0 /  |/ 
         +---------------+   +
         |  -2  -3  -2   |  /
         |  -3  -6  -3   | /
         |  -2  -3  -2   |/
         +---------------+

    
    */
    //kernel allocation.
    *kernel = (float*)malloc(27*sizeof(float));

    if (! *kernel)
    {
        return;
    }

    //first plane
    (*kernel)[0] = 2;
    (*kernel)[1] = 3;
    (*kernel)[2] = 2;

    (*kernel)[3] = 3;
    (*kernel)[4] = 6;
    (*kernel)[5] = 3;

    (*kernel)[6] = 2;
    (*kernel)[7] = 3;
    (*kernel)[8] = 2;

    //second plane
    (*kernel)[9] = 0;
    (*kernel)[10] = 0;
    (*kernel)[11] = 0;

    (*kernel)[12] = 0;
    (*kernel)[13] = 0;
    (*kernel)[14] = 0;

    (*kernel)[15] = 0;
    (*kernel)[16] = 0;
    (*kernel)[17] = 0;

    //third plane
    (*kernel)[18] =  -2;
    (*kernel)[19] =  -3;
    (*kernel)[20] =  -2;

    (*kernel)[21] =  -3;
    (*kernel)[22] =  -6;
    (*kernel)[23] =  -3;

    (*kernel)[24] =  -2;
    (*kernel)[25] =  -3;
    (*kernel)[26] =  -2;
}


template<typename dtype>
__global__ void sobel_filter_kernel_2d(dtype* image, float* output,
                                         float* dev_kernel_horizontal, float* dev_kernel_vertical,
                                         int idz, int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        float temp_vertical;
        float temp_horizontal;

        convolution2d(image + idz * xsize * ysize, &temp_horizontal, dev_kernel_horizontal, idx, idy, xsize, ysize, 3, 3);
        convolution2d(image + idz * xsize * ysize, &temp_vertical, dev_kernel_vertical, idx, idy, xsize, ysize, 3, 3);

        output[idz * xsize * ysize + idx * ysize + idy] = temp_horizontal * temp_horizontal + temp_vertical * temp_vertical;
        output[idz * xsize * ysize + idx * ysize + idy] = (float)sqrtf(output[idz * xsize * ysize + idx * ysize + idy]);
    }
}

template<typename dtype>
__global__ void sobel_filter_kernel_3d(dtype* image, float* output,
                                         float* dev_kernel_horizontal, float* dev_kernel_vertical, float* dev_kernel_depth,
                                         int xsize, int ysize, int depth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < xsize && idy < ysize && idz < depth)
    {
        float temp_vertical;
        float temp_horizontal;
        float temp_depth;

        convolution3d(image, &temp_horizontal, dev_kernel_horizontal, idx, idy, idz, xsize, ysize, depth, 3, 3, 3);
        convolution3d(image, &temp_vertical, dev_kernel_vertical, idx, idy, idz, xsize, ysize, depth, 3, 3, 3);
        convolution3d(image, &temp_depth, dev_kernel_depth, idx, idy, idz, xsize, ysize, depth, 3, 3, 3);

        output[idz * xsize * ysize + idx * ysize + idy] = temp_horizontal * temp_horizontal + temp_vertical * temp_vertical + temp_depth * temp_depth;
        output[idz * xsize * ysize + idx * ysize + idy] = (float)sqrtf(output[idz * xsize * ysize + idx * ysize + idy]);
    }
}

template __global__ void sobel_filter_kernel_2d<int>(int* image, float* output, float* dev_kernel_horizontal, float* dev_kernel_vertical, int idz,int xsize, int ysize,int zsize);
template __global__ void sobel_filter_kernel_2d<float>(float* image, float* output, float* dev_kernel_horizontal, float* dev_kernel_vertical, int idz,int xsize, int ysize,int zsize);

template __global__ void sobel_filter_kernel_3d<int>(int* image, float* output, float* dev_kernel_horizontal, float* dev_kernel_vertical, float* dev_kernel_depth, int xsize, int ysize, int depth);
template __global__ void sobel_filter_kernel_3d<float>(float* image, float* output, float* dev_kernel_horizontal, float* dev_kernel_vertical, float* dev_kernel_depth, int xsize, int ysize, int depth);



template<typename dtype>
void sobel_filtering(dtype* image, float* output, int xsize, int ysize, int zsize, bool type)
{

    dtype* dev_image;
    float* dev_output;
    cudaMalloc((void**)&dev_image, xsize * ysize * zsize * sizeof(dtype));
    cudaMalloc((void**)&dev_output, xsize * ysize * zsize * sizeof(float));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(dtype), cudaMemcpyHostToDevice);

    if (type == false)
    {
        float* kernel_horizontal;
        get_sobel_horizontal_kernel_2d(&kernel_horizontal);

        float* kernel_vertical;
        get_sobel_vertical_kernel_2d(&kernel_vertical);

        float* dev_kernel_horizontal;
        cudaMalloc((void**)&dev_kernel_horizontal, 9 * sizeof(float));
        cudaMemcpy(dev_kernel_horizontal, kernel_horizontal, 9 * sizeof(float), cudaMemcpyHostToDevice);

        float* dev_kernel_vertical;
        cudaMalloc((void**)&dev_kernel_vertical, 9 * sizeof(float));
        cudaMemcpy(dev_kernel_vertical, kernel_vertical, 9 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((xsize + blockSize.y - 1) / blockSize.y,(ysize + blockSize.x - 1) / blockSize.x );

        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < zsize; ++k) {
            sobel_filter_kernel_2d<<<gridSize, blockSize>>>(dev_image, dev_output, dev_kernel_horizontal, dev_kernel_vertical, k, xsize, ysize, zsize);
        }
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

        cudaFree(dev_kernel_horizontal);
        cudaFree(dev_kernel_vertical);    
    }

    else
    {
        float* kernel_horizontal;
        get_sobel_horizontal_kernel_3d(&kernel_horizontal);

        float* kernel_vertical;
        get_sobel_vertical_kernel_3d(&kernel_vertical);

        float* kernel_depth;
        get_sobel_depth_kernel_3d(&kernel_depth); // Corrected function name

        float* dev_kernel_horizontal;
        cudaMalloc((void**)&dev_kernel_horizontal, 27 * sizeof(float));
        cudaMemcpy(dev_kernel_horizontal, kernel_horizontal, 27 * sizeof(float), cudaMemcpyHostToDevice);

        float* dev_kernel_vertical;
        cudaMalloc((void**)&dev_kernel_vertical, 27 * sizeof(float));
        cudaMemcpy(dev_kernel_vertical, kernel_vertical, 27 * sizeof(float), cudaMemcpyHostToDevice);

        float* dev_kernel_depth;
        cudaMalloc((void**)&dev_kernel_depth, 27 * sizeof(float));
        cudaMemcpy(dev_kernel_depth, kernel_depth, 27 * sizeof(float), cudaMemcpyHostToDevice);


        dim3 blockSize(8, 8, 8);
        dim3 gridSize( (xsize + blockSize.y - 1) / blockSize.y, (ysize + blockSize.x - 1) / blockSize.x, (zsize + blockSize.z - 1) / blockSize.z);

        auto start = std::chrono::high_resolution_clock::now();

        sobel_filter_kernel_3d<<<gridSize, blockSize>>>(dev_image, dev_output,
                                                         dev_kernel_horizontal, dev_kernel_vertical, dev_kernel_depth,
                                                         xsize, ysize, zsize);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl; 

        cudaFree(dev_kernel_horizontal);
        cudaFree(dev_kernel_vertical);
        cudaFree(dev_kernel_depth);
    }

    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
}

// Explicit instantiation
template void sobel_filtering<float>(float* image, float* output, int xsize, int ysize, int zsize, bool type);
template void sobel_filtering<int>(int* image, float* output, int xsize, int ysize, int zsize, bool type);
template void sobel_filtering<unsigned int>(unsigned int* image, float* output, int xsize, int ysize, int zsize, bool type);



/*

int main()
{
    int xsize = 1024/2;
    int ysize = 1024/2;
    int zsize = 1024/2;

    static int* image;
    image = (int*)malloc(zsize*xsize*ysize*sizeof(int));

    static int* output;
    output = (int*)malloc(zsize*xsize*ysize*sizeof(int));

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
                    image[k * xsize * ysize + i * ysize + j] = -1;
                }
                
        
                output[k * xsize * ysize + i * ysize + j] = 0;
            }
        }

    }


    sobel_filtering(image,output,xsize,ysize,zsize, false);
    
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