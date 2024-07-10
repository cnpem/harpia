#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include"../src/data_structures/set.h"
#include"ccl.h"
/*
    based on: https://github.com/FolkeV/CUDA_CCL
*/

__device__ bool HasBit(int bitmask, int bit)
{
    return (bitmask & (1 << bit)) != 0;
}

__global__ void Initialization(int* block_labels, int label_step, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

    if (idx < xsize && idy < ysize)
    {
        int block_index = idy * label_step + idx;
        block_labels[block_index] = block_index;
    }
}

__global__ void Merge(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

    if (idx < xsize && idy < ysize)
    {
        int label_index = idy * label_step + idx;
        int image_index = idy * image_step + idx;

        int bit = 0;

        if (image[image_index] == 1)
        {
            bit |= 0x777;
        }

        if (image[image_index + 1] == 1)
        {
            bit |= (0x777 << 1);
        }

        if (image[image_index + image_step] == 1)
        {
            bit |= (0x777 << 4);
        }

        if (bit > 0)
        {
            if (HasBit(bit, 0) && image[image_index - image_step - 1])
            {
                union_gpu(block_labels, label_index, label_index - 2 * label_step - 2);
            }

            if ((HasBit(bit, 1) && image[image_index - image_step]) ||
                (HasBit(bit, 2) && image[image_index - image_step + 1]))
            {
                union_gpu(block_labels, label_index, label_index - 2 * label_step);
            }

            if (HasBit(bit, 3) && image[image_index - image_step + 2])
            {
                union_gpu(block_labels, label_index, label_index - 2 * label_step + 2);
            }

            if ((HasBit(bit, 4) && image[image_index - 1]) ||
                (HasBit(bit, 8) && image[image_index + image_step - 1]))
            {
                union_gpu(block_labels, label_index, label_index - 2);
            }

        }

    }

}

__global__ void CompressionLabels(int* block_labels, int label_step, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

    if (idx < xsize && idy < ysize) 
    {
        int block_index = idy * label_step + idx;
        inline_Compress(block_labels, block_index);
    }

}

__global__ void FinalLabeling(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

    if (idx < xsize && idy < ysize)
    {
        int block_index = idy * label_step + idx;
        int image_index = idy * image_step + idx;

        int label = block_labels[block_index] + 1;

        block_labels[block_index] = label * image[image_index];
        block_labels[block_index + 1] = label * image[image_index + 1];
        block_labels[block_index + label_step] = label * image[image_index + image_step];
        block_labels[block_index + label_step + 1] = label * image[image_index + image_step + 1];
    }

}


void connectedComponents(int* image, int* output, int xsize, int ysize)
{
    int* dev_image;
    int* dev_output;
    int step = xsize;

    cudaMalloc(&dev_image, xsize * ysize * sizeof(int));
    cudaMalloc(&dev_output, xsize * ysize * sizeof(int));

    cudaMemcpy(dev_image, image, xsize * ysize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, (ysize + blockDim.y - 1) / blockDim.y);

    Initialization<<<gridDim, blockDim>>>(dev_output,  step, xsize, ysize);

    Merge<<<gridDim, blockDim>>>(dev_image, dev_output, step, step, xsize, ysize);

    CompressionLabels<<<gridDim, blockDim>>>(dev_output, step, xsize, ysize);

    FinalLabeling<<<gridDim, blockDim>>>(dev_image, dev_output, step, step, xsize, ysize);

    cudaMemcpy(output, dev_output, xsize * ysize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(dev_output);


}

/*
int main() {
    const int xsize = 8;
    const int ysize = 8;

    int output[xsize*ysize];

    int image[xsize*ysize] = 
    {
         1, 1, 0, 0, 1, 1, 0, 0,
         1, 1, 0, 0, 2, 0, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 3, 1, 0, 0,
         0, 0, 0, 1, 3, 1, 1, 0,
         0, 0, 1, 3, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 0, 1, 0,
         1, 1, 0, 0, 1, 1, 0, 0
    };

    connectedComponents(image,output,xsize,ysize);

    for (int i = 0; i < ysize; i++) {
        for (int j = 0; j < xsize; j++) {
            std::cout << output[i*xsize +j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/