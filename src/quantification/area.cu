#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/quantification/area.h"

__device__ void isArea(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize, int ysize, int zsize)
{
    int image_index = idz * xsize * ysize + idy * xsize + idx;
    int counter_index = image[image_index];

    atomicAdd(&counter[counter_index], 1);
}


__device__ void isSurface(int* image, unsigned int* counter, int idx, int idy, int idz, int xsize, int ysize, int zsize)
{
    int p0 = idz * xsize * ysize + idy * xsize + idx;
    int p0_counter = image[p0];

    // All borders are perimeters (this is sufficient for this kernel format).
    if (idy - 1 < 0 || idx - 1 < 0 || idz - 1 < 0 || idy + 1 >= ysize || idx + 1 >= xsize || idz + 1 >= zsize)
    {
        atomicAdd(&counter[p0_counter], 1);
        return;
    }
    

    // Define the dimensions of the kernel
    const int depth_kernel = 3;
    const int rows_kernel = 3;
    const int cols_kernel = 3;

    // Iterate over the 26 neighbors
    for (int l = 0; l < depth_kernel; ++l)
    {

        for (int m = 0; m < rows_kernel; ++m)
        {

            for (int n = 0; n < cols_kernel; ++n)
            {

                // Compute the position with respect to the center of the kernel
                int neighbor_idz = idz - depth_kernel / 2 + l;
                int neighbor_idy = idy - rows_kernel / 2 + m;
                int neighbor_idx = idx - cols_kernel / 2 + n;

                // Skip the center voxel itself
                if (l == depth_kernel / 2 && m == rows_kernel / 2 && n == cols_kernel / 2) 
                {
                    continue;
                }

                // Check for boundary conditions
                if (neighbor_idx >= 0 && neighbor_idx < xsize &&
                    neighbor_idy >= 0 && neighbor_idy < ysize &&
                    neighbor_idz >= 0 && neighbor_idz < zsize)
                {
                    int neighbor_p = neighbor_idz * xsize * ysize + neighbor_idy * xsize + neighbor_idx;

                    if (image[p0] != image[neighbor_p])
                    {
                        atomicAdd(&counter[p0_counter], 1);
                        return;
                    }

                }

            }

        }

    }

}

__global__ void area_counter(int* image, unsigned int* counter, int idz, int xsize, int ysize, int zsize)
{
    // To compute the area, we just need to make the accumulated sum.
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);

    if (idx < xsize && idy < ysize)
    {
        isArea(image, counter, idx, idy, idz, xsize, ysize, zsize);
    }

}

__global__ void surface_area_counter(int* image, unsigned int* counter, int idz, int xsize, int ysize, int zsize)
{

    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int idy = (threadIdx.y + blockIdx.y * blockDim.y);

    if (idx < xsize && idy < ysize)
    {
        isSurface(image, counter, idx, idy, idz, xsize, ysize, zsize);
    }

}

void area(int* image, unsigned int* output, int xsize, int ysize, int zsize, bool type)
{
    int* dev_image;
    unsigned int* dev_output;

    cudaMalloc(&dev_image, xsize * ysize * zsize * sizeof(int));
    cudaMalloc(&dev_output, xsize * ysize * zsize * sizeof(unsigned int));

    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_output, 0, xsize * ysize * zsize * sizeof(unsigned int));  // Initialize output array to zero

    dim3 blockDim(32, 32);
    dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y);

    if (type == false)
    {
        // Computes the area of a 2D object in each slice.
        for (int idz = 0; idz < zsize; idz++)
        {
            area_counter<<<gridDim, blockDim>>>(dev_image, dev_output, idz, xsize, ysize, zsize);
        }
    }


    else
    {
        // Computes the surface area of a 3D object.

        for (int idz = 0; idz < zsize; idz++)
        {
            surface_area_counter<<<gridDim, blockDim>>>(dev_image, dev_output,idz, xsize, ysize, zsize);
        }
        
        
    }

    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);
}

/*
int main()
{
    const int xsize = 8;
    const int ysize = 8;
    const int zsize = 8;

    unsigned int output[xsize * ysize * zsize] = {0};

    int image[xsize * ysize * zsize] = 
    {
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
    };

    area(image, output, xsize, ysize, zsize, 1);

    for (int i = 0; i < ysize; i++) {
        for (int j = 0; j < xsize; j++) {
            std::cout << output[i*xsize +j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/