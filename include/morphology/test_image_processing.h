#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "morphology.h"

// Function prototypes

template<typename dtype>
void showImage2D(dtype *hostImage, const int xsize, const int ysize, const std::string title);

template<typename dtype>
void showImage3D(dtype *hostImage, const int xsize, const int ysize, int zsize, const std::string title);

template<typename dtype, typename dtype2>
void morphology2DopenCV(dtype *hostImage, dtype *hostOutput, 
                   const int kernel_xsize, const int kernel_ysize, 
                   const int xsize, const int ysize, dtype2 operation);

template<typename dtype, typename dtype2>
void morphology3DopenCV(dtype *hostImage, dtype *hostOutput, 
                   const int kernel_xsize, const int kernel_ysize,
                   const int xsize, const int ysize, const int zsize, dtype2 operation);


#endif // IMAGE_PROCESSING_H