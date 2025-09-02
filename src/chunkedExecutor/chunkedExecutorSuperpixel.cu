#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "../../include/morphology/cuda_helper.h"

#include <omp.h>
#include <vector>
#include <limits>
#include <algorithm>

// Merge pooled chunk results into final output with multiple statistics support
inline void mergeSuperpixelChunksWithStats(
    const std::vector<int>& h_count,
    const std::vector<double>& h_sum,
    const std::vector<float>& h_min,
    const std::vector<float>& h_max,
    int nsuperpixels,
    int base_features,
    int nchunks,
    float* hostOutput,      // shape: (nsuperpixels, nfeatures)
    int nfeatures,          // Total features including all statistics
    int stats_per_feature,
    bool writeMean,
    bool writeMin,
    bool writeMax
) {
    #pragma omp parallel for collapse(2)
    for (int spx = 0; spx < nsuperpixels; ++spx) {
        for (int base_feat = 0; base_feat < base_features; ++base_feat) {
            int total_count = 0;
            double total_sum = 0.0;
            float global_min = std::numeric_limits<float>::max();
            float global_max = std::numeric_limits<float>::lowest();

            // NOTE: Buffers are [chunk][feature][spx] => linearized as:
            // idx = c * base_features * nsuperpixels + base_feat * nsuperpixels + spx
            for (int c = 0; c < nchunks; ++c) {
                size_t idx = (size_t)c * base_features * nsuperpixels + (size_t)base_feat * nsuperpixels + spx;
                if (h_count[idx] > 0) {
                    total_count += h_count[idx];
                    total_sum += h_sum[idx];
                    global_min = std::min(global_min, h_min[idx]);
                    global_max = std::max(global_max, h_max[idx]);
                }
            }

            // Final output is [superpixel][feature]
            size_t output_base_idx = (size_t)spx * nfeatures + (size_t)base_feat * stats_per_feature;
            
            int stat_idx = 0;
            if (writeMean) {
                hostOutput[output_base_idx + stat_idx] = (total_count > 0)
                    ? static_cast<float>(total_sum / total_count)
                    : 0.0f;
                stat_idx++;
            }
            if (writeMin) {
                hostOutput[output_base_idx + stat_idx] = (total_count > 0)
                    ? global_min
                    : 0.0f;
                stat_idx++;
            }
            if (writeMax) {
                hostOutput[output_base_idx + stat_idx] = (total_count > 0)
                    ? global_max
                    : 0.0f;
                stat_idx++;
            }
        }
    }
}


template <typename Func, typename dtype, typename... Args>
void chunkedExecutorSuperpixelFeatures(Func func, int ncopies, const float safetyMargin, int ngpus, 
                             dtype* image, int* superpixel, float* output, 
                             const int xsize, const int ysize, const int zsize,
                             int nsuperpixels, int nfeatures, bool mean, bool min, bool max,
                             const int verbose, Args... args) {

    dtype* i_ref = image;
    int* s_ref = superpixel;

    // Calculate how many statistics per feature
    int stats_per_feature = 0;
    stats_per_feature += (mean ? 1 : 0);
    stats_per_feature += (min ? 1 : 0);
    stats_per_feature += (max ? 1 : 0);
    
    // Calculate base features (nfeatures includes all statistics)
    if (nfeatures % stats_per_feature != 0) {
    throw std::runtime_error("Invalid nfeatures: not divisible by number of statistics per feature.");
}
    int base_features = nfeatures / stats_per_feature;

    // Get memory allocated by the func
    int sliceSize = xsize * ysize;
    size_t sliceBytes = static_cast<size_t>(sliceSize) * sizeof(dtype) * ncopies;

    // Get number of available devices
    int ngpus_available;
    CHECK(cudaGetDeviceCount(&ngpus_available));
    
    // Adjust number of gpus to available gpus
    if (ngpus_available < 1){
        if(verbose){
            printf("No gpus available. Cannot execute operations on the gpu.");
        }
        return;
    } else if ((ngpus_available < ngpus) || (ngpus < 1)){
        if(verbose){
            printf("Number of gpus adjusted to maximum available gpus: %d.", ngpus_available);
        }
        ngpus = ngpus_available;
    }

    // Get free memory on the GPU with less memory in bytes
    size_t freeBytes, totalBytes, freeGpuBytes;
    CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

    for (int i = 1; i < ngpus; i++) {
        cudaSetDevice(i);
        CHECK(cudaMemGetInfo(&freeGpuBytes, &totalBytes));
        if (freeGpuBytes < freeBytes) {
            freeBytes = freeGpuBytes;
        }
    }

    // How many slices fit in the GPU?
    if (safetyMargin > 1 || safetyMargin < 0) {
        fprintf(stderr,
                "Error: GPU %.2f memory occupancy is invalid. Choose a value between 0 and 1.\n",
                safetyMargin);
        return;
    }
    int chunkSize = static_cast<int>(freeBytes * safetyMargin / sliceBytes);
    int nchunks = (zsize + chunkSize - 1) / chunkSize;
    
    if(verbose){
        printf("Chunks %d, stats per feature %d, base features %d .", nchunks, stats_per_feature, base_features);
    }
    
    // CASE: Not even one slice fits GPU memory
    if (chunkSize == 0) {
        fprintf(stderr,
                "Error: Not enough memory to fit even one slice. Adjust slice size or free up memory.\n");
        return;
    } else if (chunkSize >= zsize) {

        if (verbose) {
            printf("Chunk size (%d) is larger than or equal to zsize (%d). No chunking needed.\n", chunkSize, zsize);
        }
        
        // Allocate host buffers for pooling results - size for base features only
        size_t buffer_size = (size_t)nsuperpixels * base_features;
        std::vector<int> h_count(buffer_size);
        std::vector<double> h_sum(buffer_size);
        std::vector<float> h_min(buffer_size);
        std::vector<float> h_max(buffer_size);
        
        func(i_ref, s_ref, xsize, ysize, zsize, nsuperpixels, base_features,
            h_count.data(),
            h_sum.data(),
            h_min.data(),
            h_max.data(),
            mean, min, max,
            verbose, args...);

        mergeSuperpixelChunksWithStats(
            h_count, h_sum, h_min, h_max,
            nsuperpixels, base_features, 1, // nchunks = 1
            output, nfeatures, stats_per_feature, mean, min, max
        );

        if (verbose) {
            printf("\nFinished processing (no chunking needed)!\n");
        }
        return;
    }

    if (verbose) {
        printf("MaxChunkSize:%d zsize:%d ngpus:%d\n", chunkSize, zsize, ngpus);
    }

    // Allocate host buffers for pooling results - size for all chunks and base features
    size_t chunked_buffer_size = (size_t)nsuperpixels * base_features * nchunks;
    std::vector<int> h_count(chunked_buffer_size);
    std::vector<double> h_sum(chunked_buffer_size);
    std::vector<float> h_min(chunked_buffer_size);
    std::vector<float> h_max(chunked_buffer_size);

    int iz = 0;
    int chunkNumber = 0;
    int selectedDevice;
    
    // Process main chunks
    for (; iz <= zsize - chunkSize; iz += chunkSize) {
        selectedDevice = chunkNumber % ngpus;
        CHECK(cudaSetDevice(selectedDevice));
        cudaDeviceSynchronize();
        
        if (verbose) {
            printf("\niz:%d gpu:%d chunkNumber:%d\n", iz, selectedDevice, chunkNumber);
        }
                
        func(i_ref, s_ref, xsize, ysize, chunkSize, nsuperpixels, base_features,
            h_count.data() + (size_t)chunkNumber * nsuperpixels * base_features,
            h_sum.data()   + (size_t)chunkNumber * nsuperpixels * base_features,
            h_min.data()   + (size_t)chunkNumber * nsuperpixels * base_features,
            h_max.data()   + (size_t)chunkNumber * nsuperpixels * base_features,
            mean, min, max,
            verbose, args...);

        // Move to the next chunk    
        i_ref += chunkSize * sliceSize;  
        s_ref += chunkSize * sliceSize;  
        chunkNumber++;
    }

    // Process remaining chunk
    int remaining = zsize - iz;
    if (remaining > 0) {
        selectedDevice = chunkNumber % ngpus;
        CHECK(cudaSetDevice(selectedDevice));
        cudaDeviceSynchronize();
        
        if (verbose) {
            printf("\nremaining:%d gpu:%d chunkNumber:%d\n", remaining, selectedDevice, chunkNumber);
        }
        
        func(i_ref, s_ref, xsize, ysize, chunkSize, nsuperpixels, base_features,
            h_count.data() + (size_t)chunkNumber * nsuperpixels * base_features,
            h_sum.data()   + (size_t)chunkNumber * nsuperpixels * base_features,
            h_min.data()   + (size_t)chunkNumber * nsuperpixels * base_features,
            h_max.data()   + (size_t)chunkNumber * nsuperpixels * base_features,
            mean, min, max,
            verbose, args...);
        chunkNumber++;
    }
    
    if (verbose) {
        printf("\nFinished processing all chunks, merging superpixel stats!\n");
    }
    
    int actualChunks = chunkNumber;
    
    // Use the inline function for merging
    mergeSuperpixelChunksWithStats(
        h_count, h_sum, h_min, h_max,
        nsuperpixels, base_features, actualChunks,
        output, nfeatures, stats_per_feature, mean, min, max
    );

    if (verbose) {
        printf("\nFinished processing superpixel pooling!\n");
    }
}