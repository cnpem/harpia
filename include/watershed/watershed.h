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

void watershed3d(int* data, int* labels, int rows, int cols, int depth, int iterations);

void hierarchicalWatershed(int* data, int* labels, int rows, int cols, int levels);

void hierarchicalWatershed3d(int* data, int* labels,
                             int rows, int cols, int depth,
                             int levels);

void hierarchicalWatershed_2d_batched(int* image,
                                      int rows, int cols, int depth,
                                      int* labels,
                                      int levels,
                                      int dz);

template<typename in_dtype>
void watershed_gpu(const in_dtype* h_image, int* h_labels, int rows, int cols);

template<typename in_dtype>
void hierarchicalWatershed_gpu(const in_dtype* h_image, int* h_labels,
                               int ysize, int xsize, int levels);

template<typename in_dtype>
void hierarchicalWatershed_gpu_3d(const in_dtype* hostImage, int* hostLabels,
                                  int xsize, int ysize, int zsize, int flag_verbose, int levels, int neighborhood);

template<typename in_dtype>
void hierarchicalWatershedChunked(in_dtype* hostImage, int* hostLabels,
                                  int xsize, int ysize, int zsize,
                                  int levels, int neighborhood,
                                  float gpuMemory, int ngpus, int flag_verbose);


template<typename in_dtype>
void hierarchicalWatershedChunkedKernel(in_dtype* hostImage, int* hostLabels,
                                        int xsize, int ysize, int zsize,
                                        int levels, int neighborhood,
                                        float safetyMargin, int ngpus,
                                        int flag_verbose);

#endif // WATERSHED_H