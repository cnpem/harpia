#ifndef FRACTION_COUNTER_H
#define FRACTION_COUNTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

/**
 * @brief CUDA kernel to compute the fraction of non-zero (foreground) elements.
 *
 * Counts how many voxels in the image are non-zero, and accumulates the result in `counter`.
 *
 * @param[in] image Pointer to the input 3D image (flattened).
 * @param[out] counter Pointer to a device-side integer that accumulates the non-zero count.
 * @param[in] acumulator Accumulated value (optional offset).
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
__global__ void fraction_counter(int* image, int* counter, int acumulator,
                                 int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel to compute how many times each label appears in the volume.
 *
 * For each voxel, increments the corresponding counter bucket for the label value.
 *
 * @param[in] image Pointer to the input 3D labeled image.
 * @param[out] counter Pointer to an array where each index corresponds to a label and stores the count.
 * @param[in] acumulator Offset or normalization base (if needed).
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
__global__ void labels_fraction(int* image, int* counter, int acumulator,
                                int xsize, int ysize, int zsize);

/**
 * @brief Host wrapper function to compute the fraction of labeled voxels.
 *
 * Launches CUDA kernels to count either foreground presence or label distributions.
 *
 * @param[in] image Pointer to the input 3D image (flattened).
 * @param[out] output Pointer to the output array storing the counts or fractions.
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
void fraction(int* image, int* output, int xsize, int ysize, int zsize);

#endif  // FRACTION_COUNTER_H
