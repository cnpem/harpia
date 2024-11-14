#include <stdio.h>
#include <algorithm>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_snakes_2d.h"

// Structure to hold neighbor offsets
struct Neighbor {
    int dx, dy;
};
// Structure to hold gradient values
struct Gradient {
    float dx, dy;
};

__device__ bool compute_group_value(bool* levelSet, int centerIdx, int centerIdy, 
                                     const int xsize, const Neighbor* neighbors, 
                                     int group_size, bool isMax) {
    bool result = levelSet[centerIdy * xsize + centerIdx];  // Start with center pixel

    for (int i = 1; i < group_size; i++) {
        int newIdx = centerIdx + neighbors[i].dx;
        int newIdy = centerIdy + neighbors[i].dy;
        bool neighborVal = levelSet[newIdy * xsize + newIdx];
        
        if (isMax) {
            result = result || neighborVal;
        } else {
            result = result && neighborVal;
        }
    }
    return result;
}

__device__ void smoothing_pixel(bool* levelSet, bool* output,
                                const int xsize, const int ysize,
                                int centerIdx, int centerIdy,
                                bool isISd) {
    // Define the four groups of neighbors
    const int GROUPS = 4;
    const int GROUP_SIZE = 3;
    
    Neighbor groups[GROUPS][GROUP_SIZE] = {
        {{0,0}, {1,0}, {-1,0}},    // horizontal
        {{0,0}, {0,1}, {0,-1}},    // vertical
        {{0,0}, {1,1}, {-1,-1}},   // diagonal 1
        {{0,0}, {1,-1}, {-1,1}}    // diagonal 2
    };

    //Operate in groups
    bool val1 = compute_group_value(levelSet, centerIdx, centerIdy, xsize, groups[0], GROUP_SIZE, isISd);
    bool val2 = compute_group_value(levelSet, centerIdx, centerIdy, xsize, groups[1], GROUP_SIZE, isISd);
    bool val3 = compute_group_value(levelSet, centerIdx, centerIdy, xsize, groups[2], GROUP_SIZE, isISd);
    bool val4 = compute_group_value(levelSet, centerIdx, centerIdy, xsize, groups[3], GROUP_SIZE, isISd);

    if (isISd) {
        output[centerIdy * xsize + centerIdx] = val1 && val2 && val3 && val4;
    } else {
        output[centerIdy * xsize + centerIdx] = val1 || val2 || val3 || val4;
    }
}

__device__ void balloon_force_pixel(float* image, bool* levelSet, bool* output, const float threshold, const float balloonForce,
                                              int centerIdx, int centerIdy,
                                              const int xsize, const int ysize) {

  int centerIndex = centerIdy * xsize + centerIdx;

  if (image[centerIndex] > threshold) {
    bool result = levelSet[centerIndex]; // get a neighbour
    for (int iy = -1; iy <= 1; iy++) {
        for (int ix = -1; ix <= 1; ix++) {
                int neighborIndex = (centerIdy + iy) * xsize + (centerIdx + ix);
                
                //if (ix == 0 && iy == 0) continue;
                
                if (balloonForce > 0) {
                    // Dilation: take maximum (OR operation for binary)
                    result = result || levelSet[neighborIndex];
                } else {
                    // Erosion: take minimum (AND operation for binary)
                    result = result && levelSet[neighborIndex];
                }
        }
    }
    //update results
    output[centerIndex] = result;
    } else {
    output[centerIndex] = levelSet[centerIndex];
    }
}

__global__ void balloon_force_kernel(float* deviceImage, bool* deviceLevelSet, bool* deviceTemp, const float threshold, const float balloonForce, const int xsize,
                                    const int ysize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  // Skip border pixels and check bounds
  if (idx > 1 && idx < xsize-2 && idy > 1 && idy < ysize-2) {
    balloon_force_pixel(deviceImage, deviceLevelSet, deviceTemp, threshold, balloonForce,
                       idx, idy, xsize, ysize);
  }
}

template<typename dtype>
__device__ Gradient nabla(dtype* image, int centerIdx, int centerIdy, const int xsize, const int ysize) {
    Gradient result = {0.0f, 0.0f};

    // Calculate gradients (1/2 factor included)
    result.dx = 0.5f * (static_cast<float>(image[centerIdy * xsize + (centerIdx + 1)]) - 
                        static_cast<float>(image[centerIdy * xsize + (centerIdx - 1)]));
    result.dy = 0.5f * (static_cast<float>(image[(centerIdy + 1) * xsize + centerIdx]) - 
                        static_cast<float>(image[(centerIdy - 1) * xsize + centerIdx]));

    
    return result;
}

__device__ void attraction_force_pixel(float* gimage, bool* levelSet, bool* output,
                                       int centerIdx, int centerIdy,
                                       const int xsize, const int ysize) {
    int centerIndex = centerIdy * xsize + centerIdx;

    // Calculate gradients
    Gradient dgimage_local = nabla(gimage, centerIdx, centerIdy, xsize, ysize);
    Gradient du = nabla(levelSet, centerIdx, centerIdy, xsize, ysize);

    // Calculate dot product
    float factor = dgimage_local.dx * du.dx + dgimage_local.dy * du.dy;
    
    if (factor > 0.0f) {
        output[centerIndex] = true;  
    } else if (factor < 0.0f) {
        output[centerIndex] = false; 
    } else {
        output[centerIndex] = levelSet[centerIndex];
    }
}

__device__ void image_attachment_pixel(bool* deviceLevelSet, float* image, bool* output,
                                       float* deviceC1, float* deviceC2,
                                       float lambda1, float lambda2,
                                       int centerIdx, int centerIdy,
                                       const int xsize, const int ysize) {
    int centerIndex = centerIdy * xsize + centerIdx;
    //float c1 = *deviceC1;
    //float c2 = *deviceC2;

    // Calculate gradient
    Gradient du = nabla(deviceLevelSet, centerIdx, centerIdy, xsize, ysize);

    // Calculate absolute gradient
    float du_abs = fabsf(du.dx) + fabsf(du.dy);

    // Calculate squared differences
    float diff1 = static_cast<double>(image[centerIndex]) - static_cast<double>(*deviceC1);
    float diff2 = static_cast<double>(image[centerIndex]) - static_cast<double>(*deviceC2);
    double factor = lambda1 * diff1 * diff1 - lambda2 * diff2 * diff2;

    if (factor == 0 || du_abs < 0.1) {
        //output[centerIndex] = deviceLevelSet[centerIndex];
    } else if (factor > 0) {
        output[centerIndex] = false;
    } else {
        output[centerIndex] = true;
    }
}

// Kernel functions
__global__ void attraction_force_kernel(float* deviceImage, bool* deviceLevelSet, bool* deviceTemp,
                                      const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        attraction_force_pixel(deviceImage, deviceLevelSet, deviceTemp,
                             idx, idy, xsize, ysize);
    }
}

__global__ void smoothing_kernel(bool* initLevelSet, bool* deviceTemp,
                               const int xsize, const int ysize,
                               bool isISd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        smoothing_pixel(initLevelSet, deviceTemp, xsize, ysize,
                       idx, idy, isISd);
    }
}

// New helper function for applying smoothing repeatedly
void apply_smoothing_kernels(bool* &deviceLevelSet, bool* &deviceTemp, const int xsize, const int ysize, const int smoothing, bool &isISd, dim3 grid, dim3 block) {
    
    for (int mu = 0; mu < smoothing; mu++) {
        // Smoothing kernel is applied twice (ISdoSId or SIdoISd)
        smoothing_kernel<<<grid, block>>>(deviceLevelSet, deviceTemp, xsize, ysize, isISd);
        std::swap(deviceLevelSet, deviceTemp);

        isISd = !isISd;

        smoothing_kernel<<<grid, block>>>(deviceLevelSet, deviceTemp, xsize, ysize, isISd);
        std::swap(deviceLevelSet, deviceTemp);
    }
}

__global__ void image_attachment_kernel(bool* deviceLevelSet, float* deviceImage, bool* deviceTemp,
                                      float* deviceC1, float* deviceC2,
                                      float lambda1, float lambda2,
                                      const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        image_attachment_pixel(deviceLevelSet, deviceImage, deviceTemp,
                            deviceC1, deviceC2, lambda1, lambda2,
                            idx, idy, xsize, ysize);
    }
}

// Kernel to normalize the accumulated values
__global__ void normalize_kernel(float* deviceC1, float* deviceC2, int* deviceCount1, int* deviceCount2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Normalize C1 and C2 by their respective counts
        if (*deviceCount1 > 0) {
            *deviceC1 /= (*deviceCount1 * 1.0f);
        }
        if (*deviceCount2 > 0) {
            *deviceC2 /= (*deviceCount2 * 1.0f);
        }
    }
}

__global__ void reduce_kernel(const bool* deviceLevelSet, const float* deviceImage,
                              double* partialC1, double* partialC2,
                              int* partialCount1, int* partialCount2,
                              const int xsize, const int ysize) {
    extern __shared__ double sharedData[];
    double* sharedC1 = sharedData;
    double* sharedC2 = sharedData + blockDim.x * blockDim.y;
    int* sharedCount1 = (int*)(sharedData + 2 * blockDim.x * blockDim.y);
    int* sharedCount2 = sharedCount1 + blockDim.x * blockDim.y;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize shared memory
    sharedC1[tid] = 0.0;
    sharedC2[tid] = 0.0;
    sharedCount1[tid] = 0;
    sharedCount2[tid] = 0;

    __syncthreads();

    // Check bounds
    if (idx < xsize && idy < ysize) {
        int centerIndex = idy * xsize + idx;
        if (deviceLevelSet[centerIndex]) {
            sharedC1[tid] += static_cast<double>(deviceImage[centerIndex]);
            sharedCount1[tid] += 1;
        } else {
            sharedC2[tid] += static_cast<double>(deviceImage[centerIndex]);
            sharedCount2[tid] += 1;
        }
    }

    __syncthreads();

    // Perform reduction within the block
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedC1[tid] += sharedC1[tid + s];
            sharedC2[tid] += sharedC2[tid + s];
            sharedCount1[tid] += sharedCount1[tid + s];
            sharedCount2[tid] += sharedCount2[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        partialC1[blockIdx.x + gridDim.x * blockIdx.y] = sharedC1[0];
        partialC2[blockIdx.x + gridDim.x * blockIdx.y] = sharedC2[0];
        partialCount1[blockIdx.x + gridDim.x * blockIdx.y] = sharedCount1[0];
        partialCount2[blockIdx.x + gridDim.x * blockIdx.y] = sharedCount2[0];
    }
}

// Second kernel to finalize reduction
__global__ void final_reduce_kernel(double* partialC1, double* partialC2,
                                    int* partialCount1, int* partialCount2,
                                    float* deviceC1, float* deviceC2,
                                    int* deviceCount1, int* deviceCount2,
                                    int numElements) {
    extern __shared__ double sharedData[];
    double* sharedC1 = sharedData;
    double* sharedC2 = sharedData + blockDim.x;
    int* sharedCount1 = (int*)(sharedData + 2 * blockDim.x);
    int* sharedCount2 = sharedCount1 + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sharedC1[tid] = (i < numElements) ? partialC1[i] : 0.0;
    sharedC2[tid] = (i < numElements) ? partialC2[i] : 0.0;
    sharedCount1[tid] = (i < numElements) ? partialCount1[i] : 0;
    sharedCount2[tid] = (i < numElements) ? partialCount2[i] : 0;

    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedC1[tid] += sharedC1[tid + s];
            sharedC2[tid] += sharedC2[tid + s];
            sharedCount1[tid] += sharedCount1[tid + s];
            sharedCount2[tid] += sharedCount2[tid + s];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid == 0) {
        atomicAdd(deviceC1, static_cast<float>(sharedC1[0]));
        atomicAdd(deviceC2, static_cast<float>(sharedC2[0]));
        atomicAdd(deviceCount1, sharedCount1[0]);
        atomicAdd(deviceCount2, sharedCount2[0]);
    }
}

void launch_scalar_inside_outside_kernels(const bool* deviceLevelSet, const float* deviceImage,
                                          float* deviceC1, float* deviceC2,
                                          int* deviceCount1, int* deviceCount2,
                                          const int xsize, const int ysize,
                                          dim3 blockSize) {
    // Calculate grid size based on input block size
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);
    int numBlocks = gridSize.x * gridSize.y;

    // Allocate memory for partial results
    double *partialC1, *partialC2;
    int *partialCount1, *partialCount2;
    cudaMalloc(&partialC1, numBlocks * sizeof(double));
    cudaMalloc(&partialC2, numBlocks * sizeof(double));
    cudaMalloc(&partialCount1, numBlocks * sizeof(int));
    cudaMalloc(&partialCount2, numBlocks * sizeof(int));

    // Initialize output variables
    cudaMemset(deviceC1, 0, sizeof(float));
    cudaMemset(deviceC2, 0, sizeof(float));
    cudaMemset(deviceCount1, 0, sizeof(int));
    cudaMemset(deviceCount2, 0, sizeof(int));

    // Launch the first kernel
    int sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(double) + 2 * blockSize.x * blockSize.y * sizeof(int);
    reduce_kernel<<<gridSize, blockSize, sharedMemSize>>>(deviceLevelSet, deviceImage, partialC1, partialC2, partialCount1, partialCount2, xsize, ysize);

    // Determine grid and block size for the final reduction kernel
    int finalBlockSize = 256;
    int finalGridSize = (numBlocks + finalBlockSize - 1) / finalBlockSize;
    int finalSharedMemSize = 2 * finalBlockSize * sizeof(double) + 2 * finalBlockSize * sizeof(int);

    // Launch the second kernel
    final_reduce_kernel<<<finalGridSize, finalBlockSize, finalSharedMemSize>>>(partialC1, partialC2, partialCount1, partialCount2, deviceC1, deviceC2, deviceCount1, deviceCount2, numBlocks);
    
    cudaDeviceSynchronize();
    // Normalize the accumulated values
    normalize_kernel<<<1, 1>>>(deviceC1, deviceC2, deviceCount1, deviceCount2);

    // Free partial result memory
    cudaFree(partialC1);
    cudaFree(partialC2);
    cudaFree(partialCount1);
    cudaFree(partialCount2);
}


void morph_geodesic_active_contour(float* hostImage, bool* initLs, const int iterations, const float balloonForce, const float threshold, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose) {

    // Calculate memory size
    int size = xsize * ysize;
    size_t nBytes = size * sizeof(float);
    size_t nBytes_out = size * sizeof(bool);
    // Allocate device memory
    float *deviceImage;
    bool *deviceTemp, *deviceLevelSet;
    CHECK(cudaMalloc((float**)&deviceImage,  nBytes));
    CHECK(cudaMalloc((bool**)&deviceTemp, nBytes_out));
    CHECK(cudaMalloc((bool**)&deviceLevelSet, nBytes_out));

    // Copy input data to device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceLevelSet, initLs, nBytes_out, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceTemp, initLs, nBytes_out, cudaMemcpyHostToDevice));

    // Set up execution configuration
    dim3 block(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, 1);

    if (flag_verbose) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    bool isIsd = true;
    bool applyBalloonForce = (balloonForce != 0.0f);

    for (int iter = 0; iter < iterations; iter++) {
        //baloon force
        if (applyBalloonForce) {
            balloon_force_kernel<<<grid, block>>>(deviceImage, deviceLevelSet, deviceTemp, threshold, balloonForce, xsize,
                                    ysize);
            cudaGetLastError();
            std::swap(deviceLevelSet, deviceTemp);
        }

        //attraction_force
        attraction_force_kernel<<<grid, block>>>(deviceImage, deviceLevelSet, deviceTemp, xsize, ysize);
        cudaGetLastError();
        std::swap(deviceLevelSet, deviceTemp);


        // smoothing force
        apply_smoothing_kernels(deviceLevelSet, deviceTemp, xsize, ysize, smoothing, isIsd, grid, block);
        cudaGetLastError();
    }
    cudaDeviceSynchronize();
    // Copy result back to host
    CHECK(cudaMemcpy(hostOutput, deviceLevelSet, nBytes_out, cudaMemcpyDeviceToHost));
    // Clean up
    cudaFree(deviceImage);
    cudaFree(deviceTemp);
    cudaFree(deviceLevelSet);
}

void morph_chan_vese(float* hostImage, bool* initLs, const int iterations, const float lambda1, const float lambda2, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose) {
    // Calculate memory size
    int size = xsize * ysize;
    size_t nBytes = size * sizeof(float);
    size_t nBytes_out = size * sizeof(bool);

    // Allocate device memory
    float *deviceImage;
    bool *deviceTemp, *deviceLevelSet;
    CHECK(cudaMalloc((float**)&deviceImage, nBytes));
    CHECK(cudaMalloc((bool**)&deviceTemp, nBytes_out));
    CHECK(cudaMalloc((bool**)&deviceLevelSet, nBytes_out));

    // Copy input data to device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceLevelSet, initLs, nBytes_out, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceTemp, initLs, nBytes_out, cudaMemcpyHostToDevice));

    // Set up execution configuration
    dim3 block(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, 1);

    if (flag_verbose) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    // Allocate memory for c1, c2, and counts on the device
    float *deviceC1, *deviceC2;
    int *deviceCount1, *deviceCount2;
    CHECK(cudaMalloc(&deviceC1, sizeof(float)));
    CHECK(cudaMalloc(&deviceC2, sizeof(float)));
    CHECK(cudaMalloc(&deviceCount1, sizeof(int)));
    CHECK(cudaMalloc(&deviceCount2, sizeof(int)));

    bool isIsd = true;

    for (int iter = 0; iter < iterations; iter++) {
        // Launch the reduction kernels to calculate C1, C2, Count1, and Count2
        launch_scalar_inside_outside_kernels(deviceLevelSet, deviceImage, deviceC1, deviceC2, deviceCount1, deviceCount2, xsize, ysize, block);

        // Image attachment step using computed C1 and C2
        image_attachment_kernel<<<grid, block>>>(deviceLevelSet, deviceImage, deviceTemp,
                                      deviceC1, deviceC2,
                                      lambda1, lambda2,
                                      xsize, ysize);
        cudaGetLastError();
        
        // Swap level sets
        std::swap(deviceLevelSet, deviceTemp);

        // Smoothing force
        apply_smoothing_kernels(deviceLevelSet, deviceTemp, xsize, ysize, smoothing, isIsd, grid, block);
        cudaGetLastError();
    }

    // Copy result back to host
    CHECK(cudaMemcpy(hostOutput, deviceLevelSet, nBytes_out, cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(deviceImage);
    cudaFree(deviceTemp);
    cudaFree(deviceLevelSet);
    cudaFree(deviceC1);
    cudaFree(deviceC2);
    cudaFree(deviceCount1);
    cudaFree(deviceCount2);
}
