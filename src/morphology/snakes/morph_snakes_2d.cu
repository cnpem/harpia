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

__global__ void balloon_force_kernel(float* deviceImage, bool* deviceLevelSet, bool* deviceOutput, const float threshold, const float balloonForce, const int xsize,
                                    const int ysize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  // Skip border pixels and check bounds
  if (idx > 1 && idx < xsize-2 && idy > 1 && idy < ysize-2) {
    balloon_force_pixel(deviceImage, deviceLevelSet, deviceOutput, threshold, balloonForce,
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
                                       double* deviceC1, double* deviceC2,
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
    double diff1 = static_cast<double>(image[centerIndex]) - *deviceC1;
    double diff2 = static_cast<double>(image[centerIndex]) - *deviceC2;
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
__global__ void attraction_force_kernel(float* deviceImage, bool* deviceLevelSet, bool* deviceOutput,
                                      const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        attraction_force_pixel(deviceImage, deviceLevelSet, deviceOutput,
                             idx, idy, xsize, ysize);
    }
}

__global__ void smoothing_kernel(bool* initLevelSet, bool* deviceOutput,
                               const int xsize, const int ysize,
                               bool isISd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        smoothing_pixel(initLevelSet, deviceOutput, xsize, ysize,
                       idx, idy, isISd);
    }
}

// New helper function for applying smoothing repeatedly
void apply_smoothing_kernels(bool* &deviceInitLs, bool* &deviceOutput, const int xsize, const int ysize, const int smoothing, bool &isISd, dim3 grid, dim3 block) {
    
    for (int mu = 0; mu < smoothing; mu++) {
        // Smoothing kernel is applied twice (ISdoSId or SIdoISd)
        smoothing_kernel<<<grid, block>>>(deviceInitLs, deviceOutput, xsize, ysize, isISd);
        std::swap(deviceInitLs, deviceOutput);

        isISd = !isISd;

        smoothing_kernel<<<grid, block>>>(deviceInitLs, deviceOutput, xsize, ysize, isISd);
        std::swap(deviceInitLs, deviceOutput);
    }
}

__global__ void image_attachment_kernel(bool* deviceLevelSet, float* deviceImage, bool* deviceOutput,
                                      double* deviceC1, double* deviceC2,
                                      float lambda1, float lambda2,
                                      const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        image_attachment_pixel(deviceLevelSet, deviceImage, deviceOutput,
                            deviceC1, deviceC2, lambda1, lambda2,
                            idx, idy, xsize, ysize);
    }
}

__global__ void scalar_inside_outside_kernel(const bool* deviceLevelSet, const float* deviceImage, 
                                             double* deviceC1, double* deviceC2,
                                             int* deviceCount1, int* deviceCount2,
                                             const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    // Check bounds
    if (idx < xsize && idy < ysize) {
        int centerIndex = idy * xsize + idx;
        // Use atomic operations to accumulate values
        if (deviceLevelSet[centerIndex]) {
            atomicAdd(deviceC1, static_cast<double>(deviceImage[centerIndex]) );
            atomicAdd(deviceCount1, 1);
        } else {
            atomicAdd(deviceC2, static_cast<double>(deviceImage[centerIndex]) );
            atomicAdd(deviceCount2, 1);
        }
    }
}

// Kernel to normalize the accumulated values
__global__ void normalize_kernel(double* deviceC1, double* deviceC2, int* deviceCount1, int* deviceCount2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Normalize C1 and C2 by their respective counts
        if (*deviceCount1 > 0) {
            *deviceC1 /= (*deviceCount1 * 1.0f);
        }
        if (*deviceCount2 > 0) {
            *deviceC2 /= (*deviceCount2 * 1.0f);  // Corrected this line to use deviceCount2
        }
    }
}

// New helper function for calculating and normalizing scalar values
void apply_scalar_inside_outside(double* &deviceC1, double* &deviceC2, int* &deviceCount1, int* &deviceCount2, bool* &deviceInitLs, float* &deviceImage, dim3 grid, dim3 block, const int xsize, const int ysize) {
    // Reset c1 and c2 on the device to zero for each iteration
    cudaMemset(deviceC1, 0, sizeof(double));
    cudaMemset(deviceC2, 0, sizeof(double));
    cudaMemset(deviceCount1, 0, sizeof(int));
    cudaMemset(deviceCount2, 0, sizeof(int));
    
    // Calculate integral
    scalar_inside_outside_kernel<<<grid, block>>>(deviceInitLs, deviceImage, 
                                                  deviceC1, deviceC2,
                                                  deviceCount1, deviceCount2,
                                                  xsize, ysize);

    cudaDeviceSynchronize();
    // Normalize the accumulated values
    normalize_kernel<<<1, 1>>>(deviceC1, deviceC2, deviceCount1, deviceCount2);
    cudaDeviceSynchronize();

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
    bool *deviceOutput, *deviceInitLs;
    CHECK(cudaMalloc((float**)&deviceImage,  nBytes));
    CHECK(cudaMalloc((bool**)&deviceOutput, nBytes_out));
    CHECK(cudaMalloc((bool**)&deviceInitLs, nBytes_out));

    // Copy input data to device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceInitLs, initLs, nBytes_out, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceOutput, initLs, nBytes_out, cudaMemcpyHostToDevice));

    // Set up execution configuration
    dim3 block(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, 1);

    if (flag_verbose) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    bool isIsd = true;

    for (int iter = 0; iter < iterations; iter++) {
        //baloon force
        if (fabsf(balloonForce) > 1e-6f) {
            balloon_force_kernel<<<grid, block>>>(deviceImage, deviceInitLs, deviceOutput, threshold, balloonForce, xsize,
                                    ysize);
            cudaGetLastError();
            std::swap(deviceInitLs, deviceOutput);
        }

        //attraction_force
        attraction_force_kernel<<<grid, block>>>(deviceImage, deviceInitLs, deviceOutput, xsize, ysize);
        cudaGetLastError();
        std::swap(deviceInitLs, deviceOutput);


        // smoothing force
        apply_smoothing_kernels(deviceInitLs, deviceOutput, xsize, ysize, smoothing, isIsd, grid, block);
        cudaGetLastError();
    }
    cudaDeviceSynchronize();
    // Copy result back to host
    CHECK(cudaMemcpy(hostOutput, deviceInitLs, nBytes_out, cudaMemcpyDeviceToHost));
    // Clean up
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceInitLs);
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
    bool *deviceOutput, *deviceInitLs;
    CHECK(cudaMalloc((float**)&deviceImage, nBytes));
    CHECK(cudaMalloc((bool**)&deviceOutput, nBytes_out));
    CHECK(cudaMalloc((bool**)&deviceInitLs, nBytes_out));

    // Copy input data to device
    CHECK(cudaMemcpy(deviceImage,  hostImage, nBytes,     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceInitLs, initLs,    nBytes_out, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceOutput, initLs,    nBytes_out, cudaMemcpyHostToDevice));
    // Set up execution configuration
    dim3 block(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, 1);

    if (flag_verbose) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    bool isIsd = true;

    // Allocate memory for c1 and c2 on the device
    double *deviceC1, *deviceC2;
    CHECK(cudaMalloc(&deviceC1, sizeof(double)));
    CHECK(cudaMalloc(&deviceC2, sizeof(double)));
    // Need to normalize c1 and c2 by counting pixels
    int *deviceCount1, *deviceCount2; //for 3D it will need bigger values
    CHECK(cudaMalloc(&deviceCount1, sizeof(int)));
    CHECK(cudaMalloc(&deviceCount2, sizeof(int)));

    for (int iter = 0; iter < iterations; iter++) {

        // Apply scalar inside/outside and normalize
        apply_scalar_inside_outside(deviceC1, deviceC2, deviceCount1, deviceCount2, deviceInitLs, deviceImage, grid, block, xsize, ysize);

        image_attachment_kernel<<<grid, block>>>(deviceInitLs, deviceImage, deviceOutput,
                                      deviceC1, deviceC2,
                                      lambda1, lambda2,
                                      xsize, ysize);
        cudaGetLastError();                              
        
        std::swap(deviceInitLs, deviceOutput);

        // smoothing force
        apply_smoothing_kernels(deviceInitLs, deviceOutput, xsize, ysize, smoothing, isIsd, grid, block);
        cudaGetLastError();

    }

    // Copy result back to host
    CHECK(cudaMemcpy(hostOutput, deviceInitLs, nBytes_out, cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceInitLs);
    cudaFree(deviceC1);
    cudaFree(deviceC2);
    cudaFree(deviceCount1);
    cudaFree(deviceCount2);
}



