#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <float.h>

#include "../../include/superpixelExtraction/pooling_superpixel.h"
#include "../../include/superpixelExtraction/filters_in_device.h"
#include "../../include/morphology/cuda_helper.h"
#include "../../include/common/grid_block_sizes.h"
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

// Updated pooling kernel to handle feature indexing
__global__ void superpixel_pooling_kernel(
    const float* __restrict__ deviceImage,
    const int* __restrict__ deviceSuperPixel,
    int* __restrict__ count,
    double* __restrict__ sum,
    float* __restrict__ min_vals,
    float* __restrict__ max_vals,
    int total_size,
    int nsuperpixels,
    int feature_idx)  // Added feature index
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int spId = deviceSuperPixel[idx];
    if (spId < 0) return;

    float val = deviceImage[idx];

    // Calculate offset for this feature
    int offset = feature_idx * nsuperpixels + spId;

    atomicAdd(&count[offset], 1);
    atomicAddDouble(&sum[offset], static_cast<double>(val));
    atomicMinFloat(&min_vals[offset], val);
    atomicMaxFloat(&max_vals[offset], val);
}

// Helper function to initialize pooling buffers for a specific feature
void initializePoolingBuffersForFeature(int* d_count, double* d_sum, float* d_min, float* d_max, 
                                       int nsuperpixels, int feature_idx) {
    int offset = feature_idx * nsuperpixels;
    CHECK(cudaMemset(d_count + offset, 0, nsuperpixels * sizeof(int)));
    thrust::fill(thrust::device, d_sum + offset, d_sum + offset + nsuperpixels, 0.0);
    thrust::fill(thrust::device, d_min + offset, d_min + offset + nsuperpixels, FLT_MAX);
    thrust::fill(thrust::device, d_max + offset, d_max + offset + nsuperpixels, -FLT_MAX);
    CHECK(cudaDeviceSynchronize()); 
}
// Helper function to perform superpixel pooling for a specific feature
void performSuperpixelPooling(
    const float* d_source_image,
    const int* d_superpixel,
    int feature_idx,
    int base_features,
    int volume_size,
    int nsuperpixels,
    int* d_count,
    double* d_sum,
    float* d_min,
    float* d_max,
    int* h_count,
    double* h_sum,
    float* h_min,
    float* h_max,
    bool output_mean,
    bool output_min,
    bool output_max) {
    
    // Initialize buffers for this specific feature
    initializePoolingBuffersForFeature(d_count, d_sum, d_min, d_max, nsuperpixels, feature_idx);
    
    // Configure kernel launch parameters for linear access
    int block_size = 256;
    int grid_size = (volume_size + block_size - 1) / block_size;
    
    // Launch pooling kernel with feature index
    superpixel_pooling_kernel<<<grid_size, block_size>>>(
        d_source_image, d_superpixel, d_count, d_sum, d_min, d_max, 
        volume_size, nsuperpixels, feature_idx
    );
    CHECK(cudaDeviceSynchronize());
    
    // Calculate offset for this feature in host buffers - USE SIZE_T!
    size_t offset = (size_t)feature_idx * nsuperpixels;
    
    // Copy results to host for this feature
    CHECK(cudaMemcpy(h_count + offset, d_count + offset, 
                     nsuperpixels * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_sum + offset, d_sum + offset, 
                     nsuperpixels * sizeof(double), cudaMemcpyDeviceToHost));
    
    if (output_min) {
        CHECK(cudaMemcpy(h_min + offset, d_min + offset, 
                         nsuperpixels * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (output_max) {
        CHECK(cudaMemcpy(h_max + offset, d_max + offset, 
                         nsuperpixels * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// Main function
void superpixel_feature_extract_in_chunks(
    float* hostImage,
    int* hostSuperPixel,
    int xsize, int ysize, int zsize,
    int nsuperpixels,
    int base_features,
    int* h_count,
    double* h_sum,
    float* h_min,
    float* h_max,
    bool output_mean,
    bool output_min,
    bool output_max,
    int verbose,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern) {
    
    try {
        // Validate that at least one statistic is requested
        if (!output_mean && !output_min && !output_max) {
            throw std::invalid_argument("At least one of output_mean, output_min, or output_max must be true");
        }
        
        unsigned int volume_size = xsize * ysize * zsize;
        
        // Allocate GPU memory
        float* d_image;           // Original image (never modified)
        float* d_image_smoothed;  // Smoothed image (preserved per sigma)
        float* d_temp_image;      // Working buffer for filters
        int* d_superpixel;
        int* d_count;
        double* d_sum;
        float* d_min;
        float* d_max;
        float* d_temp_image_2 = nullptr; // Working buffer for texture features if needed

        CHECK(cudaMalloc(&d_image, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_image_smoothed, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_temp_image, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_superpixel, volume_size * sizeof(int)));
        CHECK(cudaMalloc(&d_count, base_features * nsuperpixels * sizeof(int)));      // Updated size
        CHECK(cudaMalloc(&d_sum, base_features * nsuperpixels * sizeof(double)));     // Updated size
        CHECK(cudaMalloc(&d_min, base_features * nsuperpixels * sizeof(float)));      // Updated size
        CHECK(cudaMalloc(&d_max, base_features * nsuperpixels * sizeof(float)));      // Updated size

        if (texture) {
            CHECK(cudaMalloc(&d_temp_image_2, volume_size * sizeof(float)));
        }
        
        // Copy data to GPU (only once)
        CHECK(cudaMemcpy(d_image, hostImage, volume_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_superpixel, hostSuperPixel, volume_size * sizeof(int), cudaMemcpyHostToDevice));
        
        int feature_index = 0;
        
        // Process each sigma value
        for (int i = 0; i < nsigmas; ++i) {
            float sigma = sigmas[i];
            if (verbose) std::cout << "Processing sigma = " << sigma << std::endl;
            
            // Apply Gaussian smoothing: d_image -> d_image_smoothed
            applyGaussianFilterDevice2D(d_image, d_image_smoothed, sigma, xsize, ysize, zsize);
            
            // Extract intensity features from smoothed image
            if (intensity) {
                performSuperpixelPooling(d_image_smoothed, d_superpixel, feature_index, base_features,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
                feature_index++;
            }
            
            // Extract edge features
            if (edges) {
                // Apply Prewitt filter: d_image_smoothed -> d_temp_image
                applyPrewittFilterDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize);
                
                // Pool the edge-filtered image
                performSuperpixelPooling(d_temp_image, d_superpixel, feature_index, base_features,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
                feature_index++;
            }
            
            // Extract texture features
            if (texture) {
                applyHessianEigenvaluesDevice2D(d_image_smoothed, d_temp_image, d_temp_image_2, xsize, ysize, zsize, 1);

                // Pool the first Hessian eigenvalue
                performSuperpixelPooling(d_temp_image, d_superpixel, feature_index, base_features,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
                feature_index++;

                // Pool the second Hessian eigenvalue
                performSuperpixelPooling(d_temp_image_2, d_superpixel, feature_index, base_features,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
                feature_index++;
            }

            // Extract shape index features
            if (shapeIndex) {
                applyShapeIndexDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize, 1);

                performSuperpixelPooling(d_temp_image, d_superpixel, feature_index, base_features,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
                feature_index++;
            }
            
            // Extract Local Binary Pattern features
            if (localBinaryPattern) {
                // Apply LBP: d_image_smoothed -> d_temp_image
                applyLocalBinaryPatternDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize);
                
                // Pool the LBP-filtered image
                performSuperpixelPooling(d_temp_image, d_superpixel, feature_index, base_features,
                                       volume_size, nsuperpixels, d_count, d_sum, d_min, d_max, 
                                       h_count, h_sum, h_min, h_max, output_mean, output_min, output_max);
                feature_index++;
            }
        }

        // Validate that we processed the expected number of features
        if (feature_index != base_features) {
            std::cerr << "Warning: Expected " << base_features << " features but processed " << feature_index << std::endl;
        }

        // Cleanup GPU memory
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_image_smoothed));
        CHECK(cudaFree(d_temp_image));
        CHECK(cudaFree(d_superpixel));
        CHECK(cudaFree(d_count));
        CHECK(cudaFree(d_sum));
        CHECK(cudaFree(d_min));
        CHECK(cudaFree(d_max));
        if (texture) {
            CHECK(cudaFree(d_temp_image_2));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in superpixel feature extraction: " << e.what() << std::endl;
        throw;
    }
}

void DeviceSuperpixelPooling2D(float* hostImage,
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
    int flag_verbose, 
    float gpuMemory, 
    int ngpus) {
    if (ngpus == 0) {
      throw std::runtime_error("CPU implementation is not available for DeviceSuperpixelPooling2D."
        "Please ensure a GPU is available to execute this function.");
    } else {
        int ncopies = 3;
        if (texture || shapeIndex) ncopies++; // hessian/shape index needs an extra copy
        chunkedExecutorSuperpixelFeatures(superpixel_feature_extract_in_chunks, ncopies, gpuMemory, ngpus, 
                             hostImage, hostSuperPixel, hostOutput, xsize, 
                             ysize, zsize, nsuperpixels, nfeatures, output_mean, output_min, output_max, flag_verbose, 
                             sigmas, nsigmas, intensity, edges, texture, shapeIndex, localBinaryPattern);
        }
}