#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cuda_runtime.h>
#include <iostream>

/*

    2d general convolution.

*/

template <typename in_dtype, typename out_dtype, typename kernel_dtype>
__device__ void convolution2d(in_dtype* input, out_dtype* output, kernel_dtype* kernel, int idx, int idy,
                              int xsize, int ysize, int nx, int ny) {
  double accumulation = 0.0;

  int inputX;
  int inputY;

  for (int m = 0; m < nx; m++) {
    for (int n = 0; n < ny; n++) {
      inputX = idx - nx / 2 + m;
      inputY = idy - ny / 2 + n;

      // Reflect padding
      if (inputX < 0) inputX = -inputX;
      else if (inputX >= xsize) inputX = 2 * xsize - inputX - 1;

      if (inputY < 0) inputY = -inputY;
      else if (inputY >= ysize) inputY = 2 * ysize - inputY - 1;

      accumulation = __fma_rn(kernel[m * ny + n], input[inputX * ysize + inputY], accumulation);
    }
  }

  *output = (out_dtype)accumulation;
}


/*

    3d general convolution.

*/
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
__device__ void convolution3d(in_dtype* input, out_dtype* output, kernel_dtype* kernel, int idx, int idy, int idz,
                              int xsize, int ysize, int zsize, int nx, int ny, int nz) {
  double accumulation = 0.0;

  int inputX, inputY, inputZ;

  for (unsigned int l = 0; l < nz; l++) {
    for (unsigned int m = 0; m < nx; m++) {
      for (unsigned int n = 0; n < ny; n++) {
        inputX = idx - nx / 2 + m;
        inputY = idy - ny / 2 + n;
        inputZ = idz - nz / 2 + l;

        // Reflect padding
        if (inputX < 0) inputX = -inputX;
        else if (inputX >= xsize) inputX = 2 * xsize - inputX - 1;

        if (inputY < 0) inputY = -inputY;
        else if (inputY >= ysize) inputY = 2 * ysize - inputY - 1;

        if (inputZ < 0) inputZ = -inputZ;
        else if (inputZ >= zsize) inputZ = 2 * zsize - inputZ - 1;

        unsigned int index = (inputZ * xsize * ysize) + (inputX * ysize) + inputY;
        accumulation = __fma_rn(kernel[(l * nx * ny) + (m * ny) + n], input[index], accumulation);
      }
    }
  }

  *output = (out_dtype)accumulation;
}

//version used in the chunked executor
template <typename in_dtype, typename out_dtype, typename kernel_dtype>
__device__ void convolution3d_chunked(in_dtype* paddedInput, out_dtype* output,
                                           kernel_dtype* kernel, int idx, int idy, int idz,
                                           int xsize, int ysize, int zsize,
                                           int padding_bottom, int padding_top,
                                           int nx, int ny, int nz) {
  double accumulation = 0.0;

  int inputX, inputY, inputZ;

  // Full padded size in Z
  const int paddedZsize = zsize + padding_bottom + padding_top;

  for (int l = 0; l < nz; ++l) {
    for (int m = 0; m < nx; ++m) {
      for (int n = 0; n < ny; ++n) {
        inputX = idx - nx / 2 + m;
        inputY = idy - ny / 2 + n;
        inputZ = idz - nz / 2 + l;

        // Reflect padding for X
        if (inputX < 0) inputX = -inputX;
        else if (inputX >= xsize) inputX = 2 * xsize - inputX - 1;

        // Reflect padding for Y
        if (inputY < 0) inputY = -inputY;
        else if (inputY >= ysize) inputY = 2 * ysize - inputY - 1;

        // Reflect padding for Z with extended bounds
        if (inputZ < -padding_bottom)
          inputZ = -inputZ - 2 * padding_bottom;
        else if (inputZ >= zsize + padding_top)
          inputZ = 2 * (zsize + padding_top) - inputZ - 1;

        const size_t index = static_cast<size_t>(inputZ) * xsize * ysize +
                             static_cast<size_t>(inputX) * ysize +
                             static_cast<size_t>(inputY);

        accumulation = __fma_rn(kernel[l * nx * ny + m * ny + n], paddedInput[index], accumulation);
      }
    }
  }

  *output = (out_dtype)accumulation;
}

#endif  // CONVOLUTION_H
