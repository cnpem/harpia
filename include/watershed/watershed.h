#ifndef WATERSHED_H
#define WATERSHED_H

#include<iostream>
#include<cuda_runtime.h>
/**
 * @brief Performs watershed segmentation using a union-find algorithm.
 *
 * This function applies watershed segmentation to a 2D image by
 * utilizing a union-find data structure for efficient region merging.
 * The input image data is processed to generate labeled regions.
 *
 * @param data       Pointer to the input image data (flattened 2D array).
 * @param labels     Pointer to the output label array (same size as input).
 * @param rows       Number of rows in the input image.
 * @param cols       Number of columns in the input image.
 * @param iterations Number of union-find iterations or passes to perform.
 */
void watershed(int* data, int* labels, int rows, int cols, int iterations);

#endif // WATERSHED_H