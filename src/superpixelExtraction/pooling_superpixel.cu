#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "../../include/superpixelExtraction/pooling_superpixel.h"
#include "../../include/common/grid_block_sizes.h"
#include "../../include/filters/gaussian_filter.h"
#include "../../include/filters/prewitt_filter.h"
#include "../../include/localBinaryPattern/lbp.h"
#include "../../include/common/chunkedExecutor.h"

// Custom atomicAdd for double (for older CUDA versions)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Custom atomicMin for float
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Custom atomicMax for float
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


// Optimized pooling kernel with improved memory access patterns
__global__ void superpixel_pooling_kernel(
    const float* __restrict__ deviceImage,
    const int* __restrict__ deviceSuperPixel,
    int* __restrict__ count,
    double* __restrict__ sum,
    float* __restrict__ min_vals,
    float* __restrict__ max_vals,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int spId = deviceSuperPixel[idx];
    if (spId < 0) return;

    float val = deviceImage[idx];
    float fval = static_cast<float>(val);

    atomicAdd(&count[spId], 1);
    atomicAddDouble(&sum[spId], static_cast<double>(val));
    atomicMinFloat(&min_vals[spId], fval);
    atomicMaxFloat(&max_vals[spId], fval);
}

// Helper function to initialize pooling buffers
void initializePoolingBuffers(int* d_count, double* d_sum, float* d_min, float* d_max, int nsuperpixels) {
    CHECK(cudaMemset(d_count, 0, nsuperpixels * sizeof(int)));
    thrust::fill(thrust::device, d_sum, d_sum + nsuperpixels, 0.0);
    thrust::fill(thrust::device, d_min, d_min + nsuperpixels, FLT_MAX);
    thrust::fill(thrust::device, d_max, d_max + nsuperpixels, -FLT_MAX);
}

// Helper function to perform superpixel pooling
void performSuperpixelPooling(
    const float* d_source_image,
    const int* d_superpixel,
    float* hostOutput,
    int& feature_idx,
    int nfeatures,
    int volume_size,
    int nsuperpixels,
    int* d_count,
    double* d_sum,
    float* d_min,
    float* d_max,
    std::vector<int>& h_count,
    std::vector<double>& h_sum,
    std::vector<float>& h_min,
    std::vector<float>& h_max,
    bool output_mean,
    bool output_min,
    bool output_max) {
    
    // Initialize buffers
    initializePoolingBuffers(d_count, d_sum, d_min, d_max, nsuperpixels);
    
    // Configure kernel launch parameters for linear access
    int block_size = 256;
    int grid_size = (volume_size + block_size - 1) / block_size;
    
    // Launch pooling kernel
    superpixel_pooling_kernel<<<grid_size, block_size>>>(
        d_source_image, d_superpixel, d_count, d_sum, d_min, d_max, volume_size
    );
    CHECK(cudaDeviceSynchronize());
    
    // Copy results to host
    CHECK(cudaMemcpy(h_count.data(), d_count, nsuperpixels * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_sum.data(), d_sum, nsuperpixels * sizeof(double), cudaMemcpyDeviceToHost));
    
    if (output_min) {
        CHECK(cudaMemcpy(h_min.data(), d_min, nsuperpixels * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (output_max) {
        CHECK(cudaMemcpy(h_max.data(), d_max, nsuperpixels * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Output mean values
    if (output_mean) {
        #pragma omp parallel for
        for (int i = 0; i < nsuperpixels; ++i) {
            hostOutput[i * nfeatures + feature_idx] = (h_count[i] > 0) 
                ? static_cast<float>(h_sum[i] / h_count[i]) 
                : 0.0f;
        }
        feature_idx++;
    }
    
    // Output min values
    if (output_min) {
        #pragma omp parallel for
        for (int i = 0; i < nsuperpixels; ++i) {
            hostOutput[i * nfeatures + feature_idx] = (h_count[i] > 0) 
                ? h_min[i] 
                : 0.0f;
        }
        feature_idx++;
    }
    
    // Output max values
    if (output_max) {
        #pragma omp parallel for
        for (int i = 0; i < nsuperpixels; ++i) {
            hostOutput[i * nfeatures + feature_idx] = (h_count[i] > 0) 
                ? h_max[i] 
                : 0.0f;
        }
        feature_idx++;
    }
}

// Helper function to apply Gaussian filter
void applyGaussianFilter(
    float* d_image,
    float* d_image_smoothed,
    float sigma,
    int xsize, int ysize, int zsize) {
    
    // Generate Gaussian kernel
    int ksize = static_cast<int>(std::ceil(4.0f * sigma + 0.5f));
    if (ksize % 2 == 0) ksize++; // Ensure odd kernel size
    
    double* h_kernel = nullptr;
    get_gaussian_kernel_2d(&h_kernel, ksize, ksize, sigma);
    
    double* d_kernel;
    CHECK(cudaMalloc(&d_kernel, ksize * ksize * sizeof(double)));
    CHECK(cudaMemcpy(d_kernel, h_kernel, ksize * ksize * sizeof(double), cudaMemcpyHostToDevice));
    free(h_kernel);
    
    // Configure 2D grid and block sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);
    
    // Apply filter slice by slice
    for (int k = 0; k < zsize; ++k) {
        gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(
            d_image, d_image_smoothed, d_kernel, k,
            xsize, ysize, zsize, ksize, ksize
        );
    }
    CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_kernel);
}

// Helper function to apply Prewitt filter
void applyPrewittFilter(
    float* d_image_smoothed,
    float* d_temp_image,
    int xsize, int ysize, int zsize) {
    
    float* kernelHorizontal;
    get_prewitt_horizontal_kernel_2d(&kernelHorizontal);

    float* kernelVertical;
    get_prewitt_vertical_kernel_2d(&kernelVertical);

    float* deviceKernelHorizontal;
    CHECK(cudaMalloc((void**)&deviceKernelHorizontal, 9 * sizeof(float)));
    CHECK(cudaMemcpy(deviceKernelHorizontal, kernelHorizontal, 9 * sizeof(float), cudaMemcpyHostToDevice));

    float* deviceKernelVertical;
    CHECK(cudaMalloc((void**)&deviceKernelVertical, 9 * sizeof(float)));
    CHECK(cudaMemcpy(deviceKernelVertical, kernelVertical, 9 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);

    // Apply Prewitt filter slice by slice
    for (int k = 0; k < zsize; ++k) {
        prewitt_filter_kernel_2d<<<gridSize, blockSize>>>(
            d_image_smoothed, d_temp_image,
            deviceKernelHorizontal, deviceKernelVertical, k, xsize, ysize,
          zsize);
    }

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(deviceKernelHorizontal));
    CHECK(cudaFree(deviceKernelVertical));
    free(kernelHorizontal);
    free(kernelVertical);
}

// Helper function to apply Local Binary Pattern
void applyLocalBinaryPattern(
    float* d_image_smoothed,
    float* d_temp_image,
    int xsize, int ysize, int zsize) {
    
    // Configure 2D grid and block sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, (ysize + blockSize.y - 1) / blockSize.y);
    
    // Apply LBP slice by slice
    for (int k = 0; k < zsize; ++k) {
        lbp<<<gridSize, blockSize>>>(
            d_image_smoothed, d_temp_image, xsize, ysize, zsize, k
        );
    }
    CHECK(cudaDeviceSynchronize());
}

// Main function
void superpixel_feature_extract(
    float* hostImage,
    int* hostSuperPixel,
    float* hostOutput,
    int xsize, int ysize, int zsize,
    int nsuperpixels,
    int nfeatures,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern,
    bool output_mean,
    bool output_min,
    bool output_max,
    int verbose) {
    
    try {
        // Validate that at least one statistic is requested
        if (!output_mean && !output_min && !output_max) {
            throw std::invalid_argument("At least one of output_mean, output_min, or output_max must be true");
        }
        
        int volume_size = xsize * ysize * zsize;
        
        // Allocate GPU memory
        float* d_image;           // Original image (never modified)
        float* d_image_smoothed;  // Smoothed image (preserved per sigma)
        float* d_temp_image;      // Working buffer for filters
        int* d_superpixel;
        int* d_count;
        double* d_sum;
        float* d_min;
        float* d_max;
        
        CHECK(cudaMalloc(&d_image, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_image_smoothed, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_temp_image, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_superpixel, volume_size * sizeof(int)));
        CHECK(cudaMalloc(&d_count, nsuperpixels * sizeof(int)));
        CHECK(cudaMalloc(&d_sum, nsuperpixels * sizeof(double)));
        CHECK(cudaMalloc(&d_min, nsuperpixels * sizeof(float)));
        CHECK(cudaMalloc(&d_max, nsuperpixels * sizeof(float)));
        
        // Copy data to GPU (only once)
        CHECK(cudaMemcpy(d_image, hostImage, volume_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_superpixel, hostSuperPixel, volume_size * sizeof(int), cudaMemcpyHostToDevice));
        
        // Allocate host buffers for pooling results
        std::vector<int> h_count(nsuperpixels);
        std::vector<double> h_sum(nsuperpixels);
        std::vector<float> h_min(nsuperpixels);
        std::vector<float> h_max(nsuperpixels);
        
        int feature_index = 0;
        
        // Process each sigma value
        for (int i = 0; i < nsigmas; ++i) {
            float sigma = sigmas[i];
            if (verbose) std::cout << "Processing sigma = " << sigma << std::endl;
            
            // Apply Gaussian smoothing: d_image -> d_image_smoothed
            applyGaussianFilter(d_image, d_image_smoothed, sigma, xsize, ysize, zsize);
            
            // Extract intensity features from smoothed image
            if (intensity) {
                performSuperpixelPooling(d_image_smoothed, d_superpixel, hostOutput, feature_index, nfeatures,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
            }
            
            // Extract edge features
            if (edges) {
                // Apply Prewitt filter: d_image_smoothed -> d_temp_image
                applyPrewittFilter(d_image_smoothed, d_temp_image, xsize, ysize, zsize);
                
                // Pool the edge-filtered image
                performSuperpixelPooling(d_temp_image, d_superpixel, hostOutput, feature_index, nfeatures,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
            }
            
            // Extract texture features (placeholder for future implementation)
            if (texture) {
                // TODO: Implement Hessian eigenvalue computation
                int stats_per_feature = (output_mean ? 1 : 0) + (output_min ? 1 : 0) + (output_max ? 1 : 0);
                feature_index += stats_per_feature;
            }
            
            // Extract shape index features (placeholder for future implementation)
            if (shapeIndex) {
                // TODO: Implement shape index computation
                int stats_per_feature = (output_mean ? 1 : 0) + (output_min ? 1 : 0) + (output_max ? 1 : 0);
                feature_index += stats_per_feature;
            }
            
            // Extract Local Binary Pattern features
            if (localBinaryPattern) {
                // Apply LBP: d_image_smoothed -> d_temp_image
                applyLocalBinaryPattern(d_image_smoothed, d_temp_image, xsize, ysize, zsize);
                
                // Pool the LBP-filtered image
                performSuperpixelPooling(d_temp_image, d_superpixel, hostOutput, feature_index, nfeatures,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
            }
        }

        // Cleanup GPU memory
        cudaFree(d_image);
        cudaFree(d_image_smoothed);
        cudaFree(d_temp_image);
        cudaFree(d_superpixel);
        cudaFree(d_count);
        cudaFree(d_sum);
        cudaFree(d_min);
        cudaFree(d_max);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in superpixel feature extraction: " << e.what() << std::endl;
        throw;
    }
}
