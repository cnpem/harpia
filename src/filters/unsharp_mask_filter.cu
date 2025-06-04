#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../../include/filters/unsharp_mask_filter.h"

/*
    adapted from: https://www.nv5geospatialsoftware.com/docs/unsharp_mask.html

    temp = Image - Convol ( Image, Gaussian )
    output = Image + A * temp * ( |temp| ≥ T )
*/

template <typename dtype>
void unsharp_mask_filtering(dtype* image, float* output, int xsize, int ysize, int zsize,
                            float sigma, float ammount, float threshold, bool type) {
  //gaussian filter application.
  gaussianFilterChunked(image,output,xsize,ysize,zsize,sigma,type,1,1,0.2);

  for (unsigned int idx = 0; idx < xsize; ++idx) {

    for (unsigned int idy = 0; idy < ysize; ++idy) {

      for (unsigned int idz = 0; idz < zsize; ++idz) {

        float temp;
        unsigned int index = idz * xsize * ysize + idx * ysize + idy;

        temp = image[index] - output[index];

        if (abs(temp) >= threshold) {
          output[index] = (float)(image[index] + ammount * temp);
        }

        else {
          output[index] = (float)image[index];
        }
      }
    }
  }
}

// Explicit instantiation
template void unsharp_mask_filtering<float>(float* image, float* output, int xsize, int ysize,
                                            int zsize, float sigma, float ammount, float threshold,
                                            bool type);
template void unsharp_mask_filtering<int>(int* image, float* output, int xsize, int ysize,
                                          int zsize, float sigma, float ammount, float threshold,
                                          bool type);
template void unsharp_mask_filtering<unsigned int>(unsigned int* image, float* output, int xsize,
                                                   int ysize, int zsize, float sigma, float ammount,
                                                   float threshold, bool type);


// new chunked version and appropriate version for benchmarking
// i will just apply gaussian filtering and then the thresholding
template <typename dtype>
void unsharpMaskChunked(dtype* image, float* output, int xsize, int ysize, int zsize,
                            float sigma, float ammount, float threshold, const int type3d, const int verbose, int ngpus,
                            const float safetyMargin) {
  //gaussian filter application.
  gaussianFilterChunked(image,output,xsize,ysize,zsize,sigma,type3d, verbose,ngpus,safetyMargin);

  for (unsigned int idx = 0; idx < xsize; ++idx) {

    for (unsigned int idy = 0; idy < ysize; ++idy) {

      for (unsigned int idz = 0; idz < zsize; ++idz) {

        float temp;
        unsigned int index = idz * xsize * ysize + idx * ysize + idy;

        temp = image[index] - output[index];

        if (abs(temp) >= threshold) {
          output[index] = (float)(image[index] + ammount * temp);
        }

        else {
          output[index] = (float)image[index];
        }
      }
    }
  }
}

// Explicit instantiation
template void unsharpMaskChunked<float>(float* image, float* output, int xsize, int ysize,
                                            int zsize, float sigma, float ammount, float threshold,
                                            const int type3d, const int verbose, int ngpus,
                                            const float safetyMargin);
template void unsharpMaskChunked<int>(int* image, float* output, int xsize, int ysize,
                                          int zsize, float sigma, float ammount, float threshold,
                                          const int type3d,const int verbose, int ngpus,
                                          const float safetyMargin);
template void unsharpMaskChunked<unsigned int>(unsigned int* image, float* output, int xsize,
                                              int ysize, int zsize, float sigma, float ammount,
                                              float threshold, const int type3d,const int verbose, int ngpus,
                                              const float safetyMargin);