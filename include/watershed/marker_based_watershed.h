#ifndef MARKER_BASED_WATERSHED_H
#define MARKER_BASED_WATERSHED_H

#include <iostream>
#include <climits>

#define MAX_SIZE 9000000  ///< Maximum size for the priority queue (used in watershed)

// -----------------------
// Union-Find Declarations
// -----------------------

/**
 * @brief Find the root of the set containing element x using path compression.
 * @param set The disjoint set array.
 * @param x The element to find.
 * @return Root of the set containing x.
 */
int findd(int* set, int x);

/**
 * @brief Perform union of two sets containing x and y.
 * @param set The disjoint set array.
 * @param x First element.
 * @param y Second element.
 */
void unionn(int* set, int x, int y);

/**
 * @brief Union-Find for processing marker pixels.
 * @param labels Label image.
 * @param idx x-coordinate.
 * @param idy y-coordinate.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 */
void union_find_markers(int* labels, int idx, int idy, int xsize, int ysize);

/**
 * @brief Union-Find for processing non-marker pixels.
 * @param labels Label image.
 * @param idx x-coordinate.
 * @param idy y-coordinate.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 */
void union_find_non_markers(int* labels, int idx, int idy, int xsize, int ysize);

/**
 * @brief Perform union-find based watershed label resolution after sorting.
 * @param sortedImage Pointer to the sorted intensity image.
 * @param labels Pointer to the label image.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 */
void union_find_watershed(int* sortedImage, int* labels, int xsize, int ysize);

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