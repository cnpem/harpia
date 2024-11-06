#include <stdio.h>
#include <algorithm>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/geodesic_morph_binary.h"
#include "../../../include/morphology/morph_snakes_2d.h"

// Structure to hold neighbor offsets
struct Neighbor {
    int dx, dy;
};
// Structure to hold gradient values
struct Gradient {
    float dx;
    float dy;
};

CUDA_HOSTDEV bool compute_group_value(bool* levelSet, int centerIdx, int centerIdy, 
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

CUDA_HOSTDEV void smoothing_pixel(bool* levelSet, bool* output,
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
    bool result;

    if (isISd) {
        result = val1 && val2 && val3 && val4;
    } else {
        result = val1 || val2 || val3 || val4;
    }
    output[centerIdy * xsize + centerIdx] = result;
}


template<typename dtype>
CUDA_HOSTDEV Gradient nabla(dtype* image, int centerIdx, int centerIdy, const int xsize) {
    Gradient result;

    // Calculate gradients (1/2 factor included)
    result.dx = 0.5f * (static_cast<float>(image[centerIdy * xsize + (centerIdx + 1)]) - 
                        static_cast<float>(image[centerIdy * xsize + (centerIdx - 1)]));
    result.dy = 0.5f * (static_cast<float>(image[(centerIdy + 1) * xsize + centerIdx]) - 
                        static_cast<float>(image[(centerIdy - 1) * xsize + centerIdx]));

    
    return result;
}

CUDA_HOSTDEV void attraction_force_pixel(float* gimage, bool* deviceLevelSet, bool* output,
                                       const int xsize, const int ysize,
                                       int centerIdx, int centerIdy) {
    int center_index = centerIdy * xsize + centerIdx;

    // Calculate gradients
    Gradient dgimage_local = nabla(gimage, centerIdx, centerIdy, xsize);
    Gradient du = nabla(deviceLevelSet, centerIdx, centerIdy, xsize);
    
    // Calculate dot product
    float factor = dgimage_local.dx * du.dx + dgimage_local.dy * du.dy;
    
    // Apply attraction force rule
    if (factor > 0) {
        output[center_index] = 1;
    } else if (factor < 0) {
        output[center_index] = 0;
    } else {
        output[center_index] = deviceLevelSet[center_index];
    }
}

CUDA_HOSTDEV void image_attachment_pixel(bool* deviceLevelSet, float* image, bool* output,
                                       float c1, float c2,
                                       float lambda1, float lambda2,
                                       int centerIdx, int centerIdy,
                                       const int xsize, const int ysize) {
    int center_index = centerIdy * xsize + centerIdx;
    
    // Calculate gradient
    Gradient du = nabla(deviceLevelSet, centerIdx, centerIdy, xsize);
    
    // Calculate absolute gradient
    float du_abs = fabsf(du.dx) + fabsf(du.dy);
    
    // Calculate squared differences
    float diff1 = image[center_index] - c1;
    float diff2 = image[center_index] - c2;
    float factor = du_abs * (lambda1 * diff1 * diff1 - lambda2 * diff2 * diff2);
    
    if (factor == 0) {
        output[center_index] = deviceLevelSet[center_index];
    } else if (factor > 0) {
        output[center_index] = 0;
    } else {
        output[center_index] = 1;
    }
}

// Kernel functions
__global__ void attraction_force_kernel(float* deviceGimage, bool* deviceLevelSet, bool* deviceOutput,
                                      const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Skip border pixels and check bounds
    if (idx > 0 && idx < xsize-1 && idy > 0 && idy < ysize-1) {
        attraction_force_pixel(deviceGimage, deviceLevelSet, deviceOutput,
                             xsize, ysize, idx, idy);
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
        cudaDeviceSynchronize();
        std::swap(deviceInitLs, deviceOutput);

        isISd = !isISd;
        smoothing_kernel<<<grid, block>>>(deviceInitLs, deviceOutput, xsize, ysize, isISd);
        cudaDeviceSynchronize();
        std::swap(deviceInitLs, deviceOutput);
    }
}

__global__ void image_attachment_kernel(float* deviceLevelSet, float* deviceImage, bool* deviceOutput,
                                      float* deviceC1, float* deviceC2,
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
                                             float* deviceC1, float* deviceC2,
                                             int* deviceCount1, int* deviceCount2,
                                             const int xsize, const int ysize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    // Check bounds
    if (idx < xsize && idy < ysize) {
        int center_index = idy * xsize + idx;
        // Use atomic operations to safely accumulate values
        if (deviceLevelSet[center_index]) {
            atomicAdd(deviceC1, deviceImage[center_index]);
            atomicAdd(deviceCount1, 1);
        } else {
            atomicAdd(deviceC2, deviceImage[center_index]);
            atomicAdd(deviceCount2, 1);
        }
    }
}

// Kernel to normalize the accumulated values
__global__ void normalize_kernel(float* deviceC1, float* deviceC2, int* deviceCount1, int* deviceCount2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Normalize C1 and C2 by their respective counts
        if (*deviceCount1 > 0) {
            *deviceC1 /= *deviceCount1;
        }
        if (*deviceCount2 > 0) {
            *deviceC2 /= *deviceCount2;
        }
    }
}

// New helper function for calculating and normalizing scalar values
void apply_scalar_inside_outside(float* &deviceC1, float* &deviceC2, int* &deviceCount1, int* &deviceCount2, bool* &deviceInitLs, float* &deviceImage, dim3 grid, dim3 block, const int xsize, const int ysize) {
    // Reset c1 and c2 on the device to zero for each iteration
    cudaMemset(deviceC1, 0, sizeof(float));
    cudaMemset(deviceC2, 0, sizeof(float));
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

void morph_geodesic_active_contour(bool* hostImage, bool* initLs, const int iterations, const float balloonForce, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose) {

    if (xsize <= 0 || ysize <= 0 || iterations <= 0) {
        printf("Invalid input parameters\n");
        return;
    }

    // Calculate memory size
    int size = xsize * ysize;
    size_t nBytes = size * sizeof(bool);
    size_t nBytes_out = size * sizeof(bool);
    // Allocate device memory
    bool *deviceImage;
    bool *deviceOutput, *deviceInitLs;
    CHECK(cudaMalloc((bool**)&deviceImage,  nBytes));
    CHECK(cudaMalloc((bool**)&deviceOutput, nBytes_out));
    CHECK(cudaMalloc((bool**)&deviceInitLs, nBytes_out));

    // Copy input data to device
    CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceInitLs, initLs, nBytes_out, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(deviceOutput, 0, nBytes_out));

    // Set up execution configuration
    dim3 block(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, 1);

    if (flag_verbose) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    bool isIsd = true;
    MorphOp operation;
    if (balloonForce < 0) {
        operation = EROSION;
    }
    else if (balloonForce > 0){
        operation = DILATION;
    }

    //define connectivity kernel for geodesic contour size for images of any dimension
    int kernel_xsize = (xsize > 2) ? 3 : xsize;
    int kernel_ysize = (ysize > 2) ? 3 : ysize;

    for (int iter = 0; iter < iterations; iter++) {

        //baloon force
        if (balloonForce != 0) {
            // input (initial level set), the mask for geodesic operation and the output of the operation
            geodesic_morph_grayscale_kernel<<<grid, block>>>(deviceInitLs, deviceImage, deviceOutput, xsize, ysize, 1, 0, 0,
            kernel_xsize, kernel_ysize, 1, operation);
            cudaDeviceSynchronize();
            std::swap(deviceInitLs, deviceOutput);
        }

        //attraction_force
        attraction_force_kernel<<<grid, block>>>(deviceImage, deviceInitLs, deviceOutput,
                                      xsize, ysize);
        cudaDeviceSynchronize();
        std::swap(deviceInitLs, deviceOutput);

        // smoothing force
        apply_smoothing_kernels(deviceInitLs, deviceOutput, xsize, ysize, smoothing, isIsd, grid, block);

    }

    // Copy result back to host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes_out, cudaMemcpyDeviceToHost));

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
    CHECK(cudaMemset(deviceOutput, 0, nBytes_out));
    
    // Set up execution configuration
    dim3 block(BLOCK_2D, BLOCK_2D, 1);
    dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y, 1);

    if (flag_verbose) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    bool isIsd = true;

    // Allocate memory for c1 and c2 on the device
    float *deviceC1, *deviceC2;
    CHECK(cudaMalloc(&deviceC1, sizeof(float)));
    CHECK(cudaMalloc(&deviceC2, sizeof(float)));
    // Need to normalize c1 and c2 by counting pixels
    int *deviceCount1, *deviceCount2;
    CHECK(cudaMalloc(&deviceCount1, sizeof(int)));
    CHECK(cudaMalloc(&deviceCount2, sizeof(int)));

    for (int iter = 0; iter < iterations; iter++) {

        // Apply scalar inside/outside and normalize
        apply_scalar_inside_outside(deviceC1, deviceC2, deviceCount1, deviceCount2, deviceInitLs, deviceImage, grid, block, xsize, ysize);


        cudaDeviceSynchronize();
        std::swap(deviceInitLs, deviceOutput);

        image_attachment_kernel<<<grid, block>>>(deviceInitLs, deviceImage, deviceOutput,
                                      deviceC1, deviceC2,
                                      lambda1, lambda2,
                                      xsize, ysize);
        cudaDeviceSynchronize();
        std::swap(deviceInitLs, deviceOutput);

        // smoothing force
        apply_smoothing_kernels(deviceInitLs, deviceOutput, xsize, ysize, smoothing, isIsd, grid, block);


    }

    // Copy result back to host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes_out, cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(deviceImage);
    cudaFree(deviceOutput);
    cudaFree(deviceInitLs);
    cudaFree(deviceC1);
    cudaFree(deviceC2);
    cudaFree(deviceCount1);
    cudaFree(deviceCount2);
}



