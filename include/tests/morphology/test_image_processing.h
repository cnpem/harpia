#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "../../morphology/morphology.h"

// Function prototypes

template <typename dtype>
void show_image_2D(dtype* hostImage, const int xsize, const int ysize, const std::string title);

template <typename dtype>
void show_image_3D(dtype* hostImage, const int xsize, const int ysize, int zsize,
                   const std::string title);

template <typename dtype, typename dtype2>
void morphology_2D_openCV(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int kernel_xsize, const int kernel_ysize, dtype2 operation);

template <typename dtype, typename dtype2>
void morphology_3D_openCV(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, const int kernel_xsize, const int kernel_ysize,
                          dtype2 operation);
#endif  // IMAGE_PROCESSING_H