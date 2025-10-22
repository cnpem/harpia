#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "../../include/morphology/cuda_helper.h"

template <typename Func, typename in_dtype, typename out_dtype, typename... Args>
void chunkedExecutorWatershed(Func func, int ncopies, const float safetyMargin, int ngpus, 
                              in_dtype* image, out_dtype* output, const int xsize, const int ysize, const int zsize, 
                              const int verbose, Args... args) 
{
    in_dtype* i_ref = image;
    out_dtype* o_ref = output;

    const int sliceSize = xsize * ysize;
    const size_t sliceBytes = static_cast<size_t>(sliceSize) * sizeof(in_dtype) * ncopies;

    // --- GPU setup ---
    int ngpus_available;
    CHECK(cudaGetDeviceCount(&ngpus_available));

    if (ngpus_available < 1) {
        if (verbose)
            printf("No GPUs available. Cannot execute operations on the GPU.\n");
        return;
    } else if ((ngpus_available < ngpus) || (ngpus < 1)) {
        if (verbose)
            printf("Number of GPUs adjusted to maximum available: %d.\n", ngpus_available);
        ngpus = ngpus_available;
    }

    // --- Find GPU with least available memory ---
    size_t freeBytes, totalBytes, freeGpuBytes;
    cudaSetDevice(0);
    CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    for (int i = 1; i < ngpus; i++) {
        cudaSetDevice(i);
        CHECK(cudaMemGetInfo(&freeGpuBytes, &totalBytes));
        if (freeGpuBytes < freeBytes)
            freeBytes = freeGpuBytes;
    }

    // --- Determine chunk size ---
    int chunkSize = static_cast<int>(freeBytes * safetyMargin / sliceBytes);
    if (verbose)
        printf("MaxChunkSize:%d zsize:%d ngpus:%d\n", chunkSize, zsize, ngpus);

    if (chunkSize == 0) {
        fprintf(stderr,
                "Error: Not enough memory to fit even one slice. Adjust slice size or free up memory.\n");
        return;
    } else if (chunkSize > zsize) {
        chunkSize = zsize;
        if (verbose)
            printf("ActualChunkSize:%d\n", chunkSize);
    }

    // --- Process image in chunks ---
    int iz = 0;
    int deviceCount = 0;
    int selectedDevice;
    int labelOffset = 0;  // <- global offset that accumulates max label from previous chunks

    for (; iz <= zsize - chunkSize; iz += chunkSize) {
        selectedDevice = deviceCount % ngpus;
        CHECK(cudaSetDevice(selectedDevice));
        cudaDeviceSynchronize();

        if (verbose) {
            printf("Processing chunk: iz=%d, chunkSize=%d, device=%d\n", iz, chunkSize, selectedDevice);
            printf("i_ref ptr: %p | max ptr: %p\n", (void*)i_ref, (void*)(image + zsize * sliceSize));
            printf("o_ref ptr: %p | max ptr: %p\n", (void*)o_ref, (void*)(output + zsize * sliceSize));
        }

        // --- Run watershed on current chunk ---
        func(i_ref, o_ref, xsize, ysize, chunkSize, verbose, args...);

        // --- Post-processing: add label offset and compute new max label ---
        int maxLabel = 0;
        const int numVoxels = chunkSize * sliceSize;

        // Apply offset in-place and find new max in this chunk
        for (int i = 0; i < numVoxels; ++i) {
            if (o_ref[i] > 0) {
                o_ref[i] += labelOffset;
                if (o_ref[i] > maxLabel)
                    maxLabel = o_ref[i];
            }
        }

        // Update the offset for next chunk
        labelOffset = maxLabel;
        std::cout<<"maxLabel: "<<labelOffset;

        if (verbose)
            printf("Chunk %d done. Max label: %d, Next offset: %d\n", deviceCount, maxLabel, labelOffset);

        // Advance to next chunk
        i_ref += chunkSize * sliceSize;
        o_ref += chunkSize * sliceSize;
        deviceCount += 1;
    }

    // --- Process remaining slices ---
    int remaining = zsize - iz;
    selectedDevice = deviceCount % ngpus;
    CHECK(cudaSetDevice(selectedDevice));
    cudaDeviceSynchronize();

    if (verbose)
        printf("\nremaining:%d gpu:%d deviceCount:%d\n", remaining, selectedDevice, deviceCount);

    if (remaining > 0) {
        func(i_ref, o_ref, xsize, ysize, remaining, verbose, args...);

        // Apply offset for last partial chunk
        int maxLabel = 0;
        const int numVoxels = remaining * sliceSize;
        for (int i = 0; i < numVoxels; ++i) {
            if (o_ref[i] > 0) {
                o_ref[i] += labelOffset;
                if (o_ref[i] > maxLabel)
                    maxLabel = o_ref[i];
            }
        }
        labelOffset = maxLabel;

        if (verbose)
            printf("Final chunk done. Max label: %d, Final offset: %d\n", maxLabel, labelOffset);
    }

    if (verbose)
        printf("\nFinished processing all chunks! Total labels: %d\n", labelOffset);
}