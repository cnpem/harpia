#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
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

// ============================================================================
// CSR construction kernels
// ============================================================================

// Count how many pixels belong to each superpixel
__global__ void count_superpixel_pixels_kernel(
    const int* __restrict__ labels,
    int total_size,
    int nsuperpixels,
    int* __restrict__ counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int lbl = labels[idx];
    // Labels are expected to be in [0, nsuperpixels), but keep this guard for safety
    if (lbl >= 0 && lbl < nsuperpixels) {
        atomicAdd(&counts[lbl], 1);
    }
}

// Fill CSR indices: for each label, write the positions of its pixels
__global__ void fill_superpixel_indices_kernel(
    const int* __restrict__ labels,
    int total_size,
    int nsuperpixels,
    int* __restrict__ write_offsets,  // mutable copy of offsets[0..nsuperpixels-1]
    int* __restrict__ indices         // size = total_size; all entries used when all pixels are labeled
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int lbl = labels[idx];
    if (lbl < 0 || lbl >= nsuperpixels) return;

    int pos = atomicAdd(&write_offsets[lbl], 1);
    indices[pos] = idx;
}

// ============================================================================
// CSR-based superpixel pooling kernel (no atomics in math phase)
// ============================================================================

__global__ void csr_superpixel_pooling_kernel(
    const float* __restrict__ feat,   // feature image, size = volume_size
    const int*   __restrict__ indices,
    const int*   __restrict__ offsets, // length = nsuperpixels + 1
    int nsuperpixels,
    int feature_idx,
    int* __restrict__ count,
    double* __restrict__ sum,
    float* __restrict__ min_vals,
    float* __restrict__ max_vals
) {
    int spId = blockIdx.x;
    if (spId >= nsuperpixels) return;

    int start = offsets[spId];
    int end   = offsets[spId + 1];
    int len   = end - start;

    int tid = threadIdx.x;

    // Local partials per thread
    float local_sum   = 0.0f;
    float local_min   = FLT_MAX;
    float local_max   = -FLT_MAX;
    int   local_count = 0;

    // Strided loop over this superpixel's pixels
    for (int i = tid; i < len; i += blockDim.x) {
        int pix_idx = indices[start + i];
        float v = feat[pix_idx];

        local_sum   += v;
        local_min    = fminf(local_min, v);
        local_max    = fmaxf(local_max, v);
        local_count += 1;
    }

    // Shared memory for reduction
    extern __shared__ unsigned char smem[];
    float* sh_sum   = (float*)smem;
    float* sh_min   = sh_sum + blockDim.x;
    float* sh_max   = sh_min + blockDim.x;
    int*   sh_count = (int*)(sh_max + blockDim.x);

    sh_sum[tid]   = local_sum;
    sh_min[tid]   = local_min;
    sh_max[tid]   = local_max;
    sh_count[tid] = local_count;
    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_sum[tid]   += sh_sum[tid + s];
            sh_min[tid]    = fminf(sh_min[tid],  sh_min[tid + s]);
            sh_max[tid]    = fmaxf(sh_max[tid],  sh_max[tid + s]);
            sh_count[tid] += sh_count[tid + s];
        }
        __syncthreads();
    }

    // Write final result for this superpixel + feature
    if (tid == 0) {
        int base = feature_idx * nsuperpixels + spId;

        int c = sh_count[0];
        count[base] = c;
        sum[base]   = static_cast<double>(sh_sum[0]);

        if (c > 0) {
            min_vals[base] = sh_min[0];
            max_vals[base] = sh_max[0];
        } else {
            // In your case (all pixels labeled), this should not happen.
            min_vals[base] = FLT_MAX;
            max_vals[base] = -FLT_MAX;
        }
    }
}

// ============================================================================
// Helper: perform pooling for a single feature using CSR
// ============================================================================

void performSuperpixelPooling(
    const float* d_source_image,
    const int* d_indices,
    const int* d_offsets,
    int feature_idx,
    int base_features,   // unused but kept for API compatibility
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
    bool output_max)
{
    (void)base_features;
    (void)volume_size;
    (void)output_mean;

    // Launch CSR pooling kernel: one block per superpixel
    int block_size = 256;
    int grid_size  = nsuperpixels;

    size_t shmem_size = block_size * (sizeof(float) * 3 + sizeof(int));

    csr_superpixel_pooling_kernel<<<grid_size, block_size, shmem_size>>>(
        d_source_image,
        d_indices,
        d_offsets,
        nsuperpixels,
        feature_idx,
        d_count,
        d_sum,
        d_min,
        d_max
    );
    CHECK(cudaDeviceSynchronize());

    // Copy results for this feature back to host
    size_t offset = static_cast<size_t>(feature_idx) * nsuperpixels;

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

// ============================================================================
// Main per-chunk feature extraction using CSR-based pooling
// ============================================================================

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
    bool localBinaryPattern)
{
    try {
        // Validate that at least one statistic is requested
        if (!output_mean && !output_min && !output_max) {
            throw std::invalid_argument("At least one of output_mean, output_min, or output_max must be true");
        }

        unsigned int volume_size_u = static_cast<unsigned int>(xsize) *
                                     static_cast<unsigned int>(ysize) *
                                     static_cast<unsigned int>(zsize);
        int volume_size = static_cast<int>(volume_size_u); // assuming fits in int

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

        CHECK(cudaMalloc(&d_image, volume_size_u * sizeof(float)));
        CHECK(cudaMalloc(&d_image_smoothed, volume_size_u * sizeof(float)));
        CHECK(cudaMalloc(&d_temp_image, volume_size_u * sizeof(float)));
        CHECK(cudaMalloc(&d_superpixel, volume_size_u * sizeof(int)));
        CHECK(cudaMalloc(&d_count, base_features * nsuperpixels * sizeof(int)));
        CHECK(cudaMalloc(&d_sum,   base_features * nsuperpixels * sizeof(double)));
        CHECK(cudaMalloc(&d_min,   base_features * nsuperpixels * sizeof(float)));
        CHECK(cudaMalloc(&d_max,   base_features * nsuperpixels * sizeof(float)));

        if (texture) {
            CHECK(cudaMalloc(&d_temp_image_2, volume_size_u * sizeof(float)));
        }

        // Copy data to GPU (only once per chunk)
        CHECK(cudaMemcpy(d_image,      hostImage,      volume_size_u * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_superpixel, hostSuperPixel, volume_size_u * sizeof(int),   cudaMemcpyHostToDevice));

        // --------------------------------------------------------------------
        // Build CSR structure for this chunk: offsets + indices
        // --------------------------------------------------------------------
        int* d_label_counts    = nullptr;
        int* d_offsets         = nullptr;  // size: nsuperpixels + 1
        int* d_write_offsets   = nullptr;
        int* d_indices         = nullptr;  // size: volume_size_u; all entries used

        CHECK(cudaMalloc(&d_label_counts, nsuperpixels * sizeof(int)));
        CHECK(cudaMemset(d_label_counts, 0, nsuperpixels * sizeof(int)));

        // Count pixels per superpixel
        {
            int block_size = 256;
            int grid_size  = (volume_size + block_size - 1) / block_size;

            count_superpixel_pixels_kernel<<<grid_size, block_size>>>(
                d_superpixel, volume_size, nsuperpixels, d_label_counts
            );
            CHECK(cudaDeviceSynchronize());
        }

        // Allocate offsets (nsuperpixels + 1)
        CHECK(cudaMalloc(&d_offsets, (nsuperpixels + 1) * sizeof(int)));

        // Exclusive scan on counts -> offsets[0..nsuperpixels-1]
        {
            thrust::device_ptr<int> counts_ptr(d_label_counts);
            thrust::device_ptr<int> offsets_ptr(d_offsets);

            thrust::exclusive_scan(thrust::device,
                                   counts_ptr,
                                   counts_ptr + nsuperpixels,
                                   offsets_ptr);
        }

        // Because all pixels are labeled, sum(counts) == volume_size.
        // So the last offset is exactly volume_size.
        CHECK(cudaMemcpy(d_offsets + nsuperpixels,
                         &volume_size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));

        // Allocate indices: one entry per pixel, all will be filled
        CHECK(cudaMalloc(&d_indices, volume_size_u * sizeof(int)));

        // Make a mutable copy of offsets[0..nsuperpixels-1] for atomic writes
        CHECK(cudaMalloc(&d_write_offsets, nsuperpixels * sizeof(int)));
        CHECK(cudaMemcpy(d_write_offsets, d_offsets,
                         nsuperpixels * sizeof(int),
                         cudaMemcpyDeviceToDevice));

        // Fill indices array
        {
            int block_size = 256;
            int grid_size  = (volume_size + block_size - 1) / block_size;

            fill_superpixel_indices_kernel<<<grid_size, block_size>>>(
                d_superpixel,
                volume_size,
                nsuperpixels,
                d_write_offsets,
                d_indices
            );
            CHECK(cudaDeviceSynchronize());
        }

        // We no longer need d_superpixel in the rest of this function
        CHECK(cudaFree(d_superpixel));
        d_superpixel = nullptr;

        // --------------------------------------------------------------------
        // Now run feature extraction + pooling using CSR
        // --------------------------------------------------------------------
        int feature_index = 0;

        for (int i = 0; i < nsigmas; ++i) {
            float sigma = sigmas[i];
            if (verbose) std::cout << "Processing sigma = " << sigma << std::endl;

            // Apply Gaussian smoothing: d_image -> d_image_smoothed
            applyGaussianFilterDevice2D(d_image, d_image_smoothed, sigma, xsize, ysize, zsize);

            // Intensity
            if (intensity) {
                performSuperpixelPooling(d_image_smoothed, d_indices, d_offsets,
                                         feature_index, base_features,
                                         volume_size, nsuperpixels,
                                         d_count, d_sum, d_min, d_max,
                                         h_count, h_sum, h_min, h_max,
                                         output_mean, output_min, output_max);
                feature_index++;
            }

            // Edges (Prewitt)
            if (edges) {
                applyPrewittFilterDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize);

                performSuperpixelPooling(d_temp_image, d_indices, d_offsets,
                                         feature_index, base_features,
                                         volume_size, nsuperpixels,
                                         d_count, d_sum, d_min, d_max,
                                         h_count, h_sum, h_min, h_max,
                                         output_mean, output_min, output_max);
                feature_index++;
            }

            // Texture (Hessian eigenvalues)
            if (texture) {
                applyHessianEigenvaluesDevice2D(d_image_smoothed, d_temp_image, d_temp_image_2,
                                                xsize, ysize, zsize, 1);

                // First eigenvalue
                performSuperpixelPooling(d_temp_image, d_indices, d_offsets,
                                         feature_index, base_features,
                                         volume_size, nsuperpixels,
                                         d_count, d_sum, d_min, d_max,
                                         h_count, h_sum, h_min, h_max,
                                         output_mean, output_min, output_max);
                feature_index++;

                // Second eigenvalue
                performSuperpixelPooling(d_temp_image_2, d_indices, d_offsets,
                                         feature_index, base_features,
                                         volume_size, nsuperpixels,
                                         d_count, d_sum, d_min, d_max,
                                         h_count, h_sum, h_min, h_max,
                                         output_mean, output_min, output_max);
                feature_index++;
            }

            // Shape index
            if (shapeIndex) {
                applyShapeIndexDevice2D(d_image_smoothed, d_temp_image,
                                        xsize, ysize, zsize, 1);

                performSuperpixelPooling(d_temp_image, d_indices, d_offsets,
                                         feature_index, base_features,
                                         volume_size, nsuperpixels,
                                         d_count, d_sum, d_min, d_max,
                                         h_count, h_sum, h_min, h_max,
                                         output_mean, output_min, output_max);
                feature_index++;
            }

            // Local Binary Pattern
            if (localBinaryPattern) {
                applyLocalBinaryPatternDevice2D(d_image_smoothed, d_temp_image,
                                                xsize, ysize, zsize);

                performSuperpixelPooling(d_temp_image, d_indices, d_offsets,
                                         feature_index, base_features,
                                         volume_size, nsuperpixels,
                                         d_count, d_sum, d_min, d_max,
                                         h_count, h_sum, h_min, h_max,
                                         output_mean, output_min, output_max);
                feature_index++;
            }
        }

        // Validate that we processed the expected number of features
        if (feature_index != base_features) {
            std::cerr << "Warning: Expected " << base_features
                      << " features but processed " << feature_index << std::endl;
        }

        // Cleanup GPU memory
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_image_smoothed));
        CHECK(cudaFree(d_temp_image));
        if (d_superpixel != nullptr) {
            CHECK(cudaFree(d_superpixel));
        }
        CHECK(cudaFree(d_count));
        CHECK(cudaFree(d_sum));
        CHECK(cudaFree(d_min));
        CHECK(cudaFree(d_max));
        if (texture) {
            CHECK(cudaFree(d_temp_image_2));
        }

        // CSR helpers
        CHECK(cudaFree(d_label_counts));
        CHECK(cudaFree(d_offsets));
        CHECK(cudaFree(d_write_offsets));
        CHECK(cudaFree(d_indices));

    } catch (const std::exception& e) {
        std::cerr << "Error in superpixel feature extraction: " << e.what() << std::endl;
        throw;
    }
}

// ============================================================================
// Public entry point
// ============================================================================

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
    int ngpus)
{
    if (ngpus == 0) {
        throw std::runtime_error(
            "CPU implementation is not available for DeviceSuperpixelPooling2D. "
            "Please ensure a GPU is available to execute this function."
        );
    } else {
        int ncopies = 3;
        if (texture || shapeIndex) ncopies++; // hessian/shape index needs an extra copy

        chunkedExecutorSuperpixelFeatures(
            superpixel_feature_extract_in_chunks,
            ncopies,
            gpuMemory,
            ngpus,
            hostImage,
            hostSuperPixel,
            hostOutput,
            xsize,
            ysize,
            zsize,
            nsuperpixels,
            nfeatures,
            output_mean,
            output_min,
            output_max,
            flag_verbose,
            sigmas,
            nsigmas,
            intensity,
            edges,
            texture,
            shapeIndex,
            localBinaryPattern
        );
    }
}
