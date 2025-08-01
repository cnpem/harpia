#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "../../include/superpixelExtraction/device_feat_extraction.h"
#include "../../include/superpixelExtraction/filters_in_device.h"
#include "../../include/morphology/cuda_helper.h"
#include "../../include/common/grid_block_sizes.h"
#include "../../include/common/chunkedExecutor.h"

// Main function
void device_feature_extract_chunks(
    float* hostImage,
    float* hostOutput,
    int xsize, int ysize, int zsize_chunk,
    int z_offset, size_t total_volume, int verbose,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern) {

    try {
        unsigned int chunk_volume = xsize * ysize * zsize_chunk;

        // Debug prints
        if (verbose) {
            printf("=== CHUNK DEBUG INFO ===\n");
            printf("Chunk dimensions: %dx%dx%d = %u voxels\n", xsize, ysize, zsize_chunk, chunk_volume);
            printf("z_offset: %d, total_volume: %zu\n", z_offset, total_volume);
            printf("Features enabled: intensity=%d, edges=%d, texture=%d, shapeIndex=%d, LBP=%d\n", 
                   intensity, edges, texture, shapeIndex, localBinaryPattern);
            printf("Number of sigmas: %d\n", nsigmas);
        }
        if (verbose) {
            printf("=== TOTAL_VOLUME DEBUG ===\n");
            printf("Function parameter total_volume: %zu\n", total_volume);
            printf("Should be (2052*2052*2048): %zu\n", (size_t)2052 * 2052 * 2048);
            printf("Chunk volume: %u\n", chunk_volume);
            printf("Copy size: %u floats\n", chunk_volume);
            printf("Feature 0 ends at: %u\n", chunk_volume);
            printf("Feature 1 starts at: %zu\n", (size_t)1 * total_volume);
            if (chunk_volume > total_volume) {
                printf("*** ERROR: Chunk size > total_volume! Data will overlap! ***\n");
            }
        }
        // Allocate GPU memory
        float* d_image;
        float* d_image_smoothed;
        float* d_temp_image;
        float* d_temp_image_2 = nullptr;

        CHECK(cudaMalloc(&d_image, chunk_volume * sizeof(float)));
        CHECK(cudaMalloc(&d_image_smoothed, chunk_volume * sizeof(float)));
        CHECK(cudaMalloc(&d_temp_image, chunk_volume * sizeof(float)));

        if (texture) {
            CHECK(cudaMalloc(&d_temp_image_2, chunk_volume * sizeof(float)));
            if (verbose) printf("Allocated d_temp_image_2 for texture processing\n");
        }

        // Copy input image to GPU
        CHECK(cudaMemcpy(d_image, hostImage, chunk_volume * sizeof(float), cudaMemcpyHostToDevice));
        if (verbose) printf("Copied input image to GPU\n");

        int feature_index = 0;
        size_t memory_offset;

        for (int i = 0; i < nsigmas; ++i) {
            float sigma = sigmas[i];
            if (verbose) {
                printf("\n--- SIGMA %d/%d: %.3f ---\n", i+1, nsigmas, sigma);
                printf("Starting feature_index: %d\n", feature_index);
            }

            // Apply Gaussian smoothing
            if (verbose) printf("Applying Gaussian filter...\n");
            applyGaussianFilterDevice2D(d_image, d_image_smoothed, sigma, xsize, ysize, zsize_chunk);
            cudaDeviceSynchronize();
            if (verbose) printf("Gaussian filter completed\n");

            if (intensity) {
                if (verbose) printf("Processing INTENSITY feature (index %d)\n", feature_index);
                memory_offset = (size_t)feature_index * total_volume + (size_t)z_offset * xsize * ysize;
                if (verbose) {
                    printf("  Memory copy offset: %zu\n", memory_offset);
                    printf("  Copy size: %u floats (%zu bytes)\n", chunk_volume, chunk_volume * sizeof(float));
                }
                CHECK(cudaMemcpy(hostOutput + memory_offset,
                                d_image_smoothed,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                cudaDeviceSynchronize();
                feature_index++;
                if (verbose) printf("  INTENSITY completed, new feature_index: %d\n", feature_index);
            }

            if (edges) {
                if (verbose) printf("Processing EDGES feature (index %d)\n", feature_index);
                applyPrewittFilterDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize_chunk);
                cudaDeviceSynchronize();
                
                memory_offset = (size_t)feature_index * total_volume + (size_t)z_offset * xsize * ysize;
                if (verbose) printf("  Memory copy offset: %zu\n", memory_offset);
                CHECK(cudaMemcpy(hostOutput + memory_offset,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                cudaDeviceSynchronize();
                feature_index++;
                if (verbose) printf("  EDGES completed, new feature_index: %d\n", feature_index);
            }

            if (texture) {
                if (verbose) {
                    printf("Processing TEXTURE features (starting at index %d)\n", feature_index);
                    printf("  About to call applyHessianEigenvaluesDevice2D...\n");
                    printf("  Input buffer: d_image_smoothed\n");
                    printf("  Output buffer 1: d_temp_image\n");
                    printf("  Output buffer 2: d_temp_image_2\n");
                }
                
                applyHessianEigenvaluesDevice2D(d_image_smoothed, d_temp_image, d_temp_image_2,
                                                xsize, ysize, zsize_chunk, 1);
                cudaDeviceSynchronize();
                if (verbose) printf("  Hessian computation completed\n");

                // Check for CUDA errors after Hessian
                if (verbose) {
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        printf("  *** CUDA ERROR after Hessian: %s ***\n", cudaGetErrorString(err));
                    }
                }

                // First eigenvalue
                if (verbose) printf("  Copying first eigenvalue (feature index %d)\n", feature_index);
                memory_offset = (size_t)feature_index * total_volume + (size_t)z_offset * xsize * ysize;
                if (verbose) printf("    Memory copy offset: %zu\n", memory_offset);
                CHECK(cudaMemcpy(hostOutput + memory_offset,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                                
                cudaDeviceSynchronize();
                feature_index++;
                if (verbose) printf("    First eigenvalue completed, new feature_index: %d\n", feature_index);
                
                // Second eigenvalue
                if (verbose) printf("  Copying second eigenvalue (feature index %d)\n", feature_index);
                memory_offset = (size_t)feature_index * total_volume + (size_t)z_offset * xsize * ysize;
                if (verbose) printf("    Memory copy offset: %zu\n", memory_offset);
                CHECK(cudaMemcpy(hostOutput + memory_offset,
                                d_temp_image_2,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                cudaDeviceSynchronize();
                feature_index++;
                if (verbose) printf("    Second eigenvalue completed, new feature_index: %d\n", feature_index);
                if (verbose) printf("  TEXTURE features completed\n");
            }

            if (shapeIndex) {
                if (verbose) printf("Processing SHAPE INDEX feature (index %d)\n", feature_index);
                applyShapeIndexDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize_chunk, 1);
                cudaDeviceSynchronize();
                
                memory_offset = (size_t)feature_index * total_volume + (size_t)z_offset * xsize * ysize;
                if (verbose) printf("  Memory copy offset: %zu\n", memory_offset);
                CHECK(cudaMemcpy(hostOutput + memory_offset,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                cudaDeviceSynchronize();
                feature_index++;
                if (verbose) printf("  SHAPE INDEX completed, new feature_index: %d\n", feature_index);
            }

            if (localBinaryPattern) {
                if (verbose) printf("Processing LOCAL BINARY PATTERN feature (index %d)\n", feature_index);
                applyLocalBinaryPatternDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize_chunk);
                cudaDeviceSynchronize();
                
                memory_offset = (size_t)feature_index * total_volume + (size_t)z_offset * xsize * ysize;
                if (verbose) printf("  Memory copy offset: %zu\n", memory_offset);
                CHECK(cudaMemcpy(hostOutput + memory_offset,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
                if (verbose) printf("  LOCAL BINARY PATTERN completed, new feature_index: %d\n", feature_index);
            }

            if (verbose) printf("End of sigma %d, final feature_index: %d\n", i+1, feature_index);
        }

        if (verbose) {
            printf("\n=== FINAL SUMMARY ===\n");
            printf("Total features processed: %d\n", feature_index);
            printf("Expected features: intensity=%d + edges=%d + texture=%d + shapeIndex=%d + LBP=%d = %d per sigma\n",
                   intensity ? 1 : 0, edges ? 1 : 0, texture ? 2 : 0, shapeIndex ? 1 : 0, localBinaryPattern ? 1 : 0,
                   (intensity ? 1 : 0) + (edges ? 1 : 0) + (texture ? 2 : 0) + (shapeIndex ? 1 : 0) + (localBinaryPattern ? 1 : 0));
            printf("Expected total: %d features × %d sigmas = %d\n", 
                   (intensity ? 1 : 0) + (edges ? 1 : 0) + (texture ? 2 : 0) + (shapeIndex ? 1 : 0) + (localBinaryPattern ? 1 : 0),
                   nsigmas,
                   ((intensity ? 1 : 0) + (edges ? 1 : 0) + (texture ? 2 : 0) + (shapeIndex ? 1 : 0) + (localBinaryPattern ? 1 : 0)) * nsigmas);
        }

        // Cleanup
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_image_smoothed));
        CHECK(cudaFree(d_temp_image));
        if (texture) {
            CHECK(cudaFree(d_temp_image_2));
            if (verbose) printf("Freed d_temp_image_2\n");
        }
        if (verbose) printf("GPU memory cleanup completed\n");

    } catch (const std::exception& e) {
        std::cerr << "Error in superpixel feature extraction: " << e.what() << std::endl;
        throw;
    }
}

void DeviceFeatExtraction2D(float* hostImage, float* hostOutput,
    int xsize, int ysize, int zsize,
    int nFeatures,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern,
    int flag_verbose, 
    float gpuMemory, 
    int ngpus) {
    if (ngpus == 0) {
      throw std::runtime_error("CPU implementation is not available for DeviceFeatExtraction2D."
        "Please ensure a GPU is available to execute this function.");
    } else {
        int ncopies = 3;
        if (texture) ncopies++; // hessian has two outputs
        chunkedExecutorPixelFeatures(device_feature_extract_chunks, ncopies, nFeatures, gpuMemory, ngpus, hostImage,
            hostOutput, xsize, ysize, zsize, flag_verbose, sigmas, nsigmas, intensity, edges, texture, shapeIndex, localBinaryPattern);

  }
}