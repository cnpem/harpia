#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cuda_runtime.h>
#include <iostream>

/*

    2d general convolution.

*/

template <typename dtype>
__device__ void convolution2d(dtype* input, float* output, float* kernel, int idx, int idy,
                              int xsize, int ysize, int nx, int ny) {
  float accumulation = 0;

  int inputX;
  int inputY;

  for (int m = 0; m < nx; m++) {
    for (int n = 0; n < ny; n++) {
      inputX = idx - nx / 2 + m;
      inputY = idy - ny / 2 + n;

      if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize) {
        accumulation += kernel[m * ny + n] * input[inputX * ysize + inputY];
      }

      else {
        // Reflect padding
        if (inputX < 0)
          inputX = -inputX;
        else if (inputX >= xsize)
          inputX = 2 * xsize - inputX - 1;

        if (inputY < 0)
          inputY = -inputY;
        else if (inputY >= ysize)
          inputY = 2 * ysize - inputY - 1;

        accumulation += kernel[m * ny + n] * input[inputX * ysize + inputY];
      }
    }
  }

  *output = (float)accumulation;
}

/*

    3d general convolution.

*/
template <typename dtype>
__device__ void convolution3d(dtype* input, float* output, float* kernel, int idx, int idy, int idz,
                              int xsize, int ysize, int zsize, int nx, int ny, int nz) {
  float accumulation = 0;

  int inputY;
  int inputX;
  int inputZ;

  for (unsigned int l = 0; l < nz; l++) {

    for (unsigned int m = 0; m < nx; m++) {

      for (unsigned int n = 0; n < ny; n++) {
        //this is needed to compute everything with respect to the center of the kernel.
        inputX = idx - nx / 2 + m;
        inputY = idy - ny / 2 + n;
        inputZ = idz - nz / 2 + l;

        //checks for boundaries.
        if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize && inputZ >= 0 &&
            inputZ < zsize) {


          unsigned int index = (inputZ * xsize * ysize) + (inputX * ysize) + inputY;  
          accumulation += kernel[(l * nx * ny) + (m * ny) + n] * input[index];
        }

        //make a padding function to substitute this line of code.
        else {
          // Reflect padding
          if (inputX < 0) {
            inputX = -inputX;
          }

          else if (inputX >= xsize) {
            inputX = 2 * xsize - inputX - 1;
          }

          if (inputY < 0) {
            inputY = -inputY;
          }

          else if (inputY >= ysize) {
            inputY = 2 * ysize - inputY - 1;
          }

          if (inputZ < 0) {
            inputZ = -inputZ;
          }

          else if (inputZ >= zsize) {
            inputZ = 2 * zsize - inputZ - 1;
          }

          unsigned int index = (inputZ * xsize * ysize) + (inputX * ysize) + inputY; 
          accumulation += kernel[(l * nx * ny) + (m * ny) + n] *input[index];
        }
      }
    }
  }

  *output = (float)accumulation;
}

#endif  // CONVOLUTION_H
