#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include "../../include/filters/non_local_means.h"

__device__ int reflect(int idx, int limit) {
    if (idx < 0) return -idx;
    if (idx >= limit) return 2 * limit - idx - 1;
    return idx;
}

template<typename dtype>
__device__ void get_nlmean_kernel_2d(dtype* image, double* mean, int idx, int idy, 
                                     int xsize, int ysize, int small_window, int big_window, double h, double sigma) {
    double accumulation = 0;
    double weight_sum = 0;
    const double epsilon = 1e-5;

    // Calculate patch radius
    int half_big = big_window / 2;
    int half_small = small_window / 2;

    // Loop through each "big window" reference patch
    for (int m = -half_big; m <= half_big; ++m) {
        for (int n = -half_big; n <= half_big; ++n) {
            int ref_center_x = reflect(idx + m, xsize);
            int ref_center_y = reflect(idy + n, ysize);

            double total_distance = 0;

            // Compute patch distance between target patch and current reference patch
            for (int p = -half_small; p <= half_small; ++p) {
                for (int q = -half_small; q <= half_small; ++q) {
                    // Coordinates in the target patch (relative to idx, idy)
                    int target_x = reflect(idx + p, xsize);
                    int target_y = reflect(idy + q, ysize);

                    // Corresponding coordinates in the reference patch (relative to ref_center_x, ref_center_y)
                    int ref_x = reflect(ref_center_x + p, xsize);
                    int ref_y = reflect(ref_center_y + q, ysize);

                    // Calculate squared difference for distance calculation
                    double diff = image[target_x * ysize + target_y] - image[ref_x * ysize + ref_y];
                    total_distance += diff * diff;
                }
            }

            // Apply Gaussian kernel on accumulated distance to get patch weight
            double h_adjusted = fmax(h * small_window, epsilon);
            double patch_weight = exp(-total_distance / (h_adjusted * h_adjusted));

            // Weighted accumulation of the reference patch value
            accumulation += patch_weight * image[ref_center_x * ysize + ref_center_y];
            weight_sum += patch_weight;
        }
    }

    // Normalize the mean
    *mean = (weight_sum > epsilon) ? accumulation / weight_sum : image[idx * ysize + idy];
}

template<typename dtype>
__global__ void nlmeans_filter_kernel_2d(dtype* deviceImage, double* deviceOutput, 
                                          int xsize, int ysize, int small_window, int big_window, double h, double sigma) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < xsize && idy < ysize) {
        // Calculate mean
        double mean = 0;
        get_nlmean_kernel_2d(deviceImage, &mean, idx, idy, xsize, ysize, small_window, big_window, h, sigma);
        
        // Assign the mean value to output
        deviceOutput[idx * ysize + idy] = mean;
    }
}

template __global__ void nlmeans_filter_kernel_2d<float>(float* deviceImage, double* deviceOutput, int xsize, int ysize, int small_window, int big_window, double h, double sigma);
template __global__ void nlmeans_filter_kernel_2d<int>(int* deviceImage, double* deviceOutput, int xsize, int ysize, int small_window, int big_window, double h, double sigma);
template __global__ void nlmeans_filter_kernel_2d<unsigned int>(unsigned int* deviceImage, double* deviceOutput, int xsize, int ysize, int small_window, int big_window, double h, double sigma);

template<typename dtype>
void nlmeans_filtering(dtype* hostImage, double* hostOutput, int xsize, int ysize, 
                       int small_window, int big_window, double h, double sigma) {
    dtype* deviceImage;
    double* deviceOutput;

    cudaMalloc((void**)&deviceImage, xsize * ysize * sizeof(dtype));
    cudaMalloc((void**)&deviceOutput, xsize * ysize * sizeof(double));

    cudaMemcpy(deviceImage, hostImage, xsize * ysize * sizeof(dtype), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    //auto start = std::chrono::high_resolution_clock::now();

    nlmeans_filter_kernel_2d<<<gridSize, blockSize>>>(deviceImage, deviceOutput, xsize, ysize, small_window, big_window, h, sigma);
    
    cudaDeviceSynchronize();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    cudaMemcpy(hostOutput, deviceOutput, xsize * ysize * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deviceImage);
    cudaFree(deviceOutput);
}

// Explicit instantiation for float and int
template void nlmeans_filtering<float>(float* hostImage, double* hostOutput, int xsize, int ysize, int small_window, int big_window, double h, double sigma);
template void nlmeans_filtering<int>(int* hostImage, double* hostOutput, int xsize, int ysize, int small_window, int big_window, double h, double sigma);
template void nlmeans_filtering<unsigned int>(unsigned int* hostImage, double* hostOutput, int xsize, int ysize, int small_window, int big_window, double h, double sigma);
