#include<iostream>
#include<cmath>
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include"../../include/quantification/connected_components.h"
#include"../../include/common/union_find.h"
/*
    based on: https://github.com/FolkeV/CUDA_CCL
*/
/*
    Disjoint set data structure -> Array-based implementation

        let the dijoint set be: 0 1 2 3 4 5 6 7 8 9

        and the corresponding array be of the same size such that: 
                            0 1 2 3 4 5 6 7 8 9 (index i)
                            0 1 2 3 4 5 6 7 8 9 (array value at the given index, array[i])

        then, if we perform a union operation, that is "union(2,1)"

        the array will be: 
                            0 1 2 3 4 5 6 7 8 9 (index i)
                            0 1 1 3 4 5 6 7 8 9 (array value at the given index, array[i])
*/

__global__ void Initialization2D(int* block_labels, int label_step, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

    if (idx < xsize && idy < ysize)
    {
        int block_index = idy * label_step + idx;
        block_labels[block_index] = block_index;
    }
}

__global__ void Merge2D(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize)
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

__global__ void CompressionLabels2D(int* block_labels, int label_step, int xsize, int ysize)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

    if (idx < xsize && idy < ysize) 
    {
        int block_index = idy * label_step + idx;
        inline_Compress(block_labels, block_index);
    }

}

__global__ void FinalLabeling2D(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize)
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


__global__ void Initialization3D(int* block_labels, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    int idz = (threadIdx.z + blockIdx.z * blockDim.z) * 2;

    if (idx < xsize && idy < ysize && idz < zsize) {
        int block_index = idz * zstep + idy * ystep + idx;
        block_labels[block_index] = block_index;
    }
}

__global__ void Merge3D(int* image, int* block_labels, int image_step, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    int idz = (threadIdx.z + blockIdx.z * blockDim.z) * 2;

    if (idx < xsize && idy < ysize && idz < zsize) {
        int label_index = idz * zstep + idy * ystep + idx;
        int image_index = idz * zstep + idy * image_step + idx;

        // Analyze the internal configuration of the 2x2x2 block
        int bit = 0;
        if (image[image_index]) bit |= (1 << 0);
        if (image[image_index + 1]) bit |= (1 << 1);
        if (image[image_index + image_step]) bit |= (1 << 2);
        if (image[image_index + image_step + 1]) bit |= (1 << 3);
        if (image[image_index + zstep]) bit |= (1 << 4);
        if (image[image_index + zstep + 1]) bit |= (1 << 5);
        if (image[image_index + zstep + image_step]) bit |= (1 << 6);
        if (image[image_index + zstep + image_step + 1]) bit |= (1 << 7);

        // Perform unions with 26 neighbors
        if (bit > 0) {
            // Back neighbors (-zstep)
            if (idx > 0 && idy > 0 && idz > 0 && image[image_index - zstep - image_step - 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep - 2 * label_step - 2);
            }
            if (idy > 0 && idz > 0 && image[image_index - zstep - image_step]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep - 2 * label_step);
            }
            if (idx < xsize - 1 && idy > 0 && idz > 0 && image[image_index - zstep - image_step + 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep - 2 * label_step + 2);
            }
            if (idx > 0 && idz > 0 && image[image_index - zstep - 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep - 2);
            }
            if (idz > 0 && image[image_index - zstep]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep);
            }
            if (idx < xsize - 1 && idz > 0 && image[image_index - zstep + 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep + 2);
            }
            if (idx > 0 && idy < ysize - 1 && idz > 0 && image[image_index - zstep + image_step - 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep + 2 * label_step - 2);
            }
            if (idy < ysize - 1 && idz > 0 && image[image_index - zstep + image_step]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep + 2 * label_step);
            }
            if (idx < xsize - 1 && idy < ysize - 1 && idz > 0 && image[image_index - zstep + image_step + 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * zstep + 2 * label_step + 2);
            }

            // Same-layer neighbors
            if (idx > 0 && idy > 0 && image[image_index - image_step - 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * label_step - 2);
            }
            if (idy > 0 && image[image_index - image_step]) {
                union_gpu(block_labels, label_index, label_index - 2 * label_step);
            }
            if (idx < xsize - 1 && idy > 0 && image[image_index - image_step + 1]) {
                union_gpu(block_labels, label_index, label_index - 2 * label_step + 2);
            }
            if (idx > 0 && image[image_index - 1]) {
                union_gpu(block_labels, label_index, label_index - 2);
            }
            if (idx < xsize - 1 && image[image_index + 1]) {
                union_gpu(block_labels, label_index, label_index + 2);
            }
            if (idx > 0 && idy < ysize - 1 && image[image_index + image_step - 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * label_step - 2);
            }
            if (idy < ysize - 1 && image[image_index + image_step]) {
                union_gpu(block_labels, label_index, label_index + 2 * label_step);
            }
            if (idx < xsize - 1 && idy < ysize - 1 && image[image_index + image_step + 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * label_step + 2);
            }

            // Front neighbors (+zstep)
            if (idx > 0 && idy > 0 && idz < zsize - 1 && image[image_index + zstep - image_step - 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep - 2 * label_step - 2);
            }
            if (idy > 0 && idz < zsize - 1 && image[image_index + zstep - image_step]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep - 2 * label_step);
            }
            if (idx < xsize - 1 && idy > 0 && idz < zsize - 1 && image[image_index + zstep - image_step + 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep - 2 * label_step + 2);
            }
            if (idx > 0 && idz < zsize - 1 && image[image_index + zstep - 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep - 2);
            }
            if (idz < zsize - 1 && image[image_index + zstep]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep);
            }
            if (idx < xsize - 1 && idz < zsize - 1 && image[image_index + zstep + 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep + 2);
            }
            if (idx > 0 && idy < ysize - 1 && idz < zsize - 1 && image[image_index + zstep + image_step - 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep + 2 * label_step - 2);
            }
            if (idy < ysize - 1 && idz < zsize - 1 && image[image_index + zstep + image_step]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep + 2 * label_step);
            }
            if (idx < xsize - 1 && idy < ysize - 1 && idz < zsize - 1 && image[image_index + zstep + image_step + 1]) {
                union_gpu(block_labels, label_index, label_index + 2 * zstep + 2 * label_step + 2);
            }
        }
    }
}

__global__ void CompressionLabels3D(int* block_labels, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    int idz = (threadIdx.z + blockIdx.z * blockDim.z) * 2;

    if (idx < xsize && idy < ysize && idz < zsize) {
        int block_index = idz * zstep + idy * ystep + idx;
        block_labels[block_index] = find(block_labels, block_index);
    }
}

__global__ void FinalLabeling3D(int* image, int* block_labels, int image_step, int label_step, int ystep, int zstep, int xsize, int ysize, int zsize) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int idy = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    int idz = (threadIdx.z + blockIdx.z * blockDim.z) * 2;

    if (idx < xsize && idy < ysize && idz < zsize) {
        int block_index = idz * zstep + idy * ystep + idx;
        int image_index = idz * zstep + idy * image_step + idx;

        // Get the label for the current block
        int label = block_labels[block_index] + 1;

        // Update labels for all 8 voxels in the 2x2x2 cube
        for (int z = 0; z < 2; ++z) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    int voxel_idx = (idz + z) * zstep + (idy + y) * ystep + (idx + x);
                    if ((idz + z) < zsize && (idy + y) < ysize && (idx + x) < xsize) {
                        block_labels[voxel_idx] = label * image[voxel_idx];
                    }
                }
            }
        }
    }
}

void connectedComponents(int* image, int* output, int xsize, int ysize, int zsize, bool type) {
    int* dev_image;
    int* dev_output;

    // Step sizes for 3D case
    int ystep = xsize;
    int zstep = xsize * ysize;

    // Allocate device memory
    cudaMalloc(&dev_image, xsize * ysize * zsize * sizeof(int));
    cudaMalloc(&dev_output, xsize * ysize * zsize * sizeof(int));

    // Copy input image to device
    cudaMemcpy(dev_image, image, xsize * ysize * zsize * sizeof(int), cudaMemcpyHostToDevice);

    if (type == false) 
    {
        // 2D case
        dim3 blockDim(32, 32);
        dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, 
                     (ysize + blockDim.y - 1) / blockDim.y);

        Initialization2D<<<gridDim, blockDim>>>(dev_output, xsize, xsize, ysize);
        cudaDeviceSynchronize();

        Merge2D<<<gridDim, blockDim>>>(dev_image, dev_output, xsize, xsize, xsize, ysize);
        cudaDeviceSynchronize();

        CompressionLabels2D<<<gridDim, blockDim>>>(dev_output, xsize, xsize, ysize);
        cudaDeviceSynchronize();

        FinalLabeling2D<<<gridDim, blockDim>>>(dev_image, dev_output, xsize, xsize, xsize, ysize);
        cudaDeviceSynchronize();
    } 

    else 
    {
        // 3D case
        dim3 blockDim(8, 8, 8);
        dim3 gridDim((xsize + blockDim.x - 1) / blockDim.x, 
                     (ysize + blockDim.y - 1) / blockDim.y, 
                     (zsize + blockDim.z - 1) / blockDim.z);

        Initialization3D<<<gridDim, blockDim>>>(dev_output, ystep, ystep, zstep, xsize, ysize, zsize);
        cudaDeviceSynchronize();

        Merge3D<<<gridDim, blockDim>>>(dev_image, dev_output, ystep, ystep, ystep, zstep, xsize, ysize, zsize);
        cudaDeviceSynchronize();

        CompressionLabels3D<<<gridDim, blockDim>>>(dev_output, ystep, ystep, zstep, xsize, ysize, zsize);
        cudaDeviceSynchronize();

        FinalLabeling3D<<<gridDim, blockDim>>>(dev_image, dev_output, xsize, xsize, ystep, zstep, xsize, ysize, zsize);
        cudaDeviceSynchronize();
    }

    // Copy output from device to host
    cudaMemcpy(output, dev_output, xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
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