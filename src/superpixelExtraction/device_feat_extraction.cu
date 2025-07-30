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
void device_feature_extract(
    float* hostImage,
    float* hostOutput,
    int xsize, int ysize, int zsize,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern,
    int verbose) {

    try {
        unsigned int volume_size = xsize * ysize * zsize;

        // Allocate GPU memory
        float* d_image;
        float* d_image_smoothed;
        float* d_temp_image;
        float* d_temp_image_2 = nullptr;

        CHECK(cudaMalloc(&d_image, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_image_smoothed, volume_size * sizeof(float)));
        CHECK(cudaMalloc(&d_temp_image, volume_size * sizeof(float)));

        if (texture) {
            CHECK(cudaMalloc(&d_temp_image_2, volume_size * sizeof(float)));
        }

        // Copy input image to GPU
        CHECK(cudaMemcpy(d_image, hostImage, volume_size * sizeof(float), cudaMemcpyHostToDevice));

        int feature_index = 0;

        for (int i = 0; i < nsigmas; ++i) {
            float sigma = sigmas[i];
            if (verbose) std::cout << "Processing sigma = " << sigma << std::endl;

            // Apply Gaussian smoothing
            applyGaussianFilterDevice2D(d_image, d_image_smoothed, sigma, xsize, ysize, zsize);

            if (intensity) {
                CHECK(cudaMemcpy(hostOutput + feature_index * volume_size, d_image_smoothed,
                                 volume_size * sizeof(float), cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (edges) {
                applyPrewittFilterDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize);
                CHECK(cudaMemcpy(hostOutput + feature_index * volume_size, d_temp_image,
                                 volume_size * sizeof(float), cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (texture) {
                applyHessianEigenvaluesDevice2D(d_image_smoothed, d_temp_image, d_temp_image_2,
                                                xsize, ysize, zsize, 1);
                CHECK(cudaMemcpy(hostOutput + feature_index * volume_size, d_temp_image,
                                 volume_size * sizeof(float), cudaMemcpyDeviceToHost));
                feature_index++;
                CHECK(cudaMemcpy(hostOutput + feature_index * volume_size, d_temp_image_2,
                                 volume_size * sizeof(float), cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (shapeIndex) {
                applyShapeIndexDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize, 1);
                CHECK(cudaMemcpy(hostOutput + feature_index * volume_size, d_temp_image,
                                 volume_size * sizeof(float), cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (localBinaryPattern) {
                applyLocalBinaryPatternDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize);
                CHECK(cudaMemcpy(hostOutput + feature_index * volume_size, d_temp_image,
                                 volume_size * sizeof(float), cudaMemcpyDeviceToHost));
                feature_index++;
            }
        }

        // Cleanup
        cudaFree(d_image);
        cudaFree(d_image_smoothed);
        cudaFree(d_temp_image);
        if (texture) {
            cudaFree(d_temp_image_2);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in superpixel feature extraction: " << e.what() << std::endl;
        throw;
    }
}

// Main function
void device_feature_extract_chunks(
    float* hostImage,
    float* hostOutput,
    int xsize, int ysize, int zsize_chunk,
    int z_offset, unsigned int total_volume, int verbose,
    float* sigmas,
    int nsigmas,
    bool intensity,
    bool edges,
    bool texture,
    bool shapeIndex,
    bool localBinaryPattern) {

    try {
        unsigned int chunk_volume = xsize * ysize * zsize_chunk;

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
        }

        // Copy input image to GPU
        CHECK(cudaMemcpy(d_image, hostImage, chunk_volume * sizeof(float), cudaMemcpyHostToDevice));

        int feature_index = 0;

        for (int i = 0; i < nsigmas; ++i) {
            float sigma = sigmas[i];
            if (verbose) std::cout << "Processing sigma = " << sigma << std::endl;

            // Apply Gaussian smoothing
            applyGaussianFilterDevice2D(d_image, d_image_smoothed, sigma, xsize, ysize, zsize_chunk);

            if (intensity) {
                CHECK(cudaMemcpy(hostOutput + feature_index * total_volume,
                                d_image_smoothed,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (edges) {
                applyPrewittFilterDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize_chunk);
                CHECK(cudaMemcpy(hostOutput + feature_index * total_volume,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (texture) {
                applyHessianEigenvaluesDevice2D(d_image_smoothed, d_temp_image, d_temp_image_2,
                                                xsize, ysize, zsize_chunk, 1);
                CHECK(cudaMemcpy(hostOutput + feature_index * total_volume,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
                CHECK(cudaMemcpy(hostOutput + feature_index * total_volume,
                                d_temp_image_2,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (shapeIndex) {
                applyShapeIndexDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize_chunk, 1);
                CHECK(cudaMemcpy(hostOutput + feature_index * total_volume,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
            }

            if (localBinaryPattern) {
                applyLocalBinaryPatternDevice2D(d_image_smoothed, d_temp_image, xsize, ysize, zsize_chunk);
                CHECK(cudaMemcpy(hostOutput + feature_index * total_volume,
                                d_temp_image,
                                chunk_volume * sizeof(float),
                                cudaMemcpyDeviceToHost));
                feature_index++;
            }
        }

        // Cleanup
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_image_smoothed));
        CHECK(cudaFree(d_temp_image));
        if (texture) {
            CHECK(cudaFree(d_temp_image_2));
        }

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