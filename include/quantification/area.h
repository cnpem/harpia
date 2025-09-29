#ifndef AREA_SURFACE_COUNTER_H
#define AREA_SURFACE_COUNTER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>

/**
 * @brief Device function to increment the area counter based on the current voxel.
 *
 * This function checks if the current voxel belongs to the object (e.g., foreground = 1)
 * and increments the area counter accordingly.
 *
 * @param[in] image Pointer to 3D image data on the device.
 * @param[out] counter Pointer to the counter updated on the device.
 * @param[in] idx X-coordinate.
 * @param[in] idy Y-coordinate.
 * @param[in] idz Z-coordinate.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
__device__ void isArea(int* image, unsigned int* counter, int idx, int idy, int idz,
                       int xsize, int ysize, int zsize);

/**
 * @brief Device function to increment the surface area counter based on voxel neighbors.
 *
 * This function checks if the current voxel is part of the object and has background neighbors,
 * which indicates it lies on the surface.
 *
 * @param[in] image Pointer to 3D image data on the device.
 * @param[out] counter Pointer to the counter updated on the device.
 * @param[in] idx X-coordinate.
 * @param[in] idy Y-coordinate.
 * @param[in] idz Z-coordinate.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
__device__ void isSurface(int* image, unsigned int* counter, int idx, int idy, int idz,
                          int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel to count the area (foreground voxels) of a given 2D slice.
 *
 * @param[in] image Pointer to 3D image data on the device.
 * @param[out] counter Pointer to the device counter for area.
 * @param[in] idz Index of the Z-slice to process.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
__global__ void area_counter(int* image, unsigned int* counter, int idz,
                             int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel to count the surface area of a given 2D slice.
 *
 * @param[in] image Pointer to 3D image data on the device.
 * @param[out] counter Pointer to the device counter for surface area.
 * @param[in] idz Index of the Z-slice to process.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 */
__global__ void surface_area_counter(int* image, unsigned int* counter, int idz,
                                     int xsize, int ysize, int zsize);

/**
 * @brief Host function to compute either area or surface area of a 3D binary image.
 *
 * This function dispatches either `area_counter` or `surface_area_counter` kernels
 * depending on the `type` flag.
 *
 * @param[in] image Pointer to the 3D image data on the host.
 * @param[out] output Pointer to the host-side counter result.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 * @param[in] zsize Depth of the image.
 * @param[in] type If true, compute surface area; if false, compute total area.
 */
void area(int* image, unsigned int* output, int xsize, int ysize, int zsize, bool type);

#endif  // AREA_SURFACE_COUNTER_H