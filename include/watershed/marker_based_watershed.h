#ifndef MARKER_BASED_WATERSHED_H
#define MARKER_BASED_WATERSHED_H

#include <iostream>
#include <climits>

#define MAX_SIZE 9000000  ///< Maximum size for the priority queue (used in watershed)
// -----------------------------
// Watershed Algorithm Wrappers
// -----------------------------

/**
 * @brief Perform Meyer's marker-based watershed segmentation in 2D.
 * @param R Input grayscale image.
 * @param M Marker image (predefined seeds).
 * @param bg Background marker value.
 * @param rows Image height.
 * @param cols Image width.
 */
void meyers_watershed_2d(int* R, int* M, int bg, int rows, int cols);

/**
 * @brief Perform Meyer's marker-based watershed segmentation in 3D.
 * @param R Input 3D image (flattened).
 * @param M Marker volume (flattened).
 * @param bg Background marker value.
 * @param depth Number of slices.
 * @param rows Image height.
 * @param cols Image width.
 */
void meyers_watershed_3d(int* R, int* M, int bg, int depth, int rows, int cols);

#endif // MARKER_BASED_WATERSHED_H