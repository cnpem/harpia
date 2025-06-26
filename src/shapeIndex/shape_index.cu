#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/shapeIndex/shape_index.h"
//performs finite differences with first order approximation over the edges
template<typename in_dtype>
__global__ void gradient2D(in_dtype* devImage, float* devOutput,
                           int xsize, int ysize, int zsize, int idz, int axis, float step)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize)
    {
        const unsigned int index = idz * xsize * ysize + idx * ysize + idy;
        float temp = 0.0f;

        // Handle axis 0 (x-direction)
        if (axis == 0)
        {

            if (idx == 0)
            {
                unsigned int forward = idz * xsize * ysize + (idx + step) * ysize + idy;
                temp = devImage[forward] - devImage[index];
            }

            else if (idx == xsize - step)
            {
                unsigned int back = idz * xsize * ysize + (idx - step) * ysize + idy;
                temp = devImage[index] - devImage[back];
            }

            else
            {
                unsigned int forward = idz * xsize * ysize + (idx + step) * ysize + idy;
                unsigned int back = idz * xsize * ysize + (idx - step) * ysize + idy;
                temp = 0.5f * (devImage[forward] - devImage[back]);
            }

        }

        // Handle axis 1 (y-direction)
        else if (axis == 1)
        {
            if (idy == 0)
            {
                unsigned int forward = idz * xsize * ysize + idx * ysize + (idy + step);
                temp = devImage[forward] - devImage[index];
            }

            else if (idy == ysize - step)
            {
                unsigned int back = idz * xsize * ysize + idx * ysize + (idy - step);
                temp = devImage[index] - devImage[back];
            }

            else
            {
                unsigned int forward = idz * xsize * ysize + idx * ysize + (idy + step);
                unsigned int back = idz * xsize * ysize + idx * ysize + (idy - step);
                temp = 0.5f * (devImage[forward] - devImage[back]);
            }

        }

        devOutput[index] = temp / step;
    }

}

// Dummy instantiations to ensure compilation for specific types
template __global__ void gradient2D<int>(int* devImage, float* devOutput,
                                         int xsize, int ysize, int zsize, int idz, int axis, float step);

template __global__ void gradient2D<unsigned int>(unsigned int* devImage, float* devOutput,
                                                  int xsize, int ysize, int zsize, int idz, int axis, float step);

template __global__ void gradient2D<float>(float* devImage, float* devOutput,
                                           int xsize, int ysize, int zsize, int idz, int axis, float step);


template <typename in_dtype>
void gradient(in_dtype* hostImage, float* hostOutput, int xsize, int ysize, int zsize, int axis, float step) 
{

    in_dtype* devImage;
    float* devOutput;

    cudaMalloc((void**)&devImage, xsize * ysize * zsize * sizeof(in_dtype));
    cudaMalloc((void**)&devOutput, xsize * ysize * zsize * sizeof(float));

    cudaMemcpy(devImage, hostImage, xsize * ysize * zsize * sizeof(in_dtype), cudaMemcpyHostToDevice);


    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    for (int idz = 0; idz < zsize; ++idz)
    {
        gradient2D<in_dtype><<<gridSize, blockSize>>>(devImage, devOutput, xsize, ysize, zsize, idz, axis, step );

        cudaDeviceSynchronize();
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration =
        //std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
    cudaMemcpy(hostOutput, devOutput, xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devImage);
    cudaFree(devOutput);

  }

template void gradient<int>(int* hostImage, float* hostOutput,
                            int xsize, int ysize, int zsize, int axis, float step);

template void gradient<unsigned int>(unsigned int* hostImage, float* hostOutput,
                                     int xsize, int ysize, int zsize, int axis, float step);

template void gradient<float>(float* hostImage, float* hostOutput,
                              int xsize, int ysize, int zsize, int axis, float step);