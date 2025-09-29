#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#define PI 3.141592653589793

template <typename dtype>
__device__ void get_mean_kernel_2d(dtype* image, float* mean, int idx, int idy, int xsize,
                                   int ysize, int nx, int ny) {

  int inputY;
  int inputX;

  float accumulation = 0;

  for (int m = 0; m < nx; m++) {

    for (int n = 0; n < ny; n++) {
      //this is needed to compute everything with respect to the center of the kernel.
      inputX = idx - nx / 2 + m;
      inputY = idy - ny / 2 + n;

      // Check if inputX and inputY are within bounds
      if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize) {
        accumulation += image[inputX * ysize + inputY];
      }

      //make a padding function to substitute this line of code.
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

        accumulation += image[inputX * ysize + inputY];
      }
    }
  }

  *mean = accumulation / (nx * ny);
}

template <typename dtype>
__device__ void get_mean_kernel_3d(dtype* image, float* mean, int idx, int idy, int idz, int xsize,
                                   int ysize, int zsize, int nx, int ny, int nz) {

  float accumulation = 0;

  int inputY;
  int inputX;
  int inputZ;

  for (int l = 0; l < nz; l++) {

    for (int m = 0; m < nx; m++) {

      for (int n = 0; n < ny; n++) {
        //this is needed to compute everything with respect to the center of the kernel.
        inputX = idx - nx / 2 + m;
        inputY = idy - ny / 2 + n;
        inputZ = idz - nz / 2 + l;

        if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize && inputZ >= 0 &&
            inputZ < zsize) {
          accumulation += image[(inputZ * xsize * ysize) + (inputX * ysize) + inputY];
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

          accumulation += image[(inputZ * xsize * ysize) + (inputX * ysize) + inputY];
        }
      }
    }
  }

  *mean = accumulation / (nx * ny * nz);
}

template <typename dtype>
__device__ void get_std_kernel_2d(dtype* image, float mean, float* standard_deviation, int idx,
                                  int idy, int xsize, int ysize, int nx, int ny) {

  int inputY;
  int inputX;

  float accumulation = 0;

  for (int m = 0; m < nx; m++) {

    for (int n = 0; n < ny; n++) {
      //this is needed to compute everything with respect to the center of the kernel.
      inputX = idx - nx / 2 + m;
      inputY = idy - ny / 2 + n;

      // Check if inputX and inputY are within bounds
      if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize) {
        accumulation += pow(image[inputX * ysize + inputY] - mean, 2);
      }

      //make a padding function to substitute this line of code.
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

        accumulation += pow(image[inputX * ysize + inputY] - mean, 2);
      }
    }
  }

  *standard_deviation = sqrt(accumulation / (nx * ny));
}

template <typename dtype>
__device__ void get_std_kernel_3d(dtype* image, float mean, float* standard_deviation, int idx,
                                  int idy, int idz, int xsize, int ysize, int zsize, int nx, int ny,
                                  int nz) {

  float accumulation = 0;

  int inputY;
  int inputX;
  int inputZ;

  for (int l = 0; l < nz; l++) {

    for (int m = 0; m < nx; m++) {

      for (int n = 0; n < ny; n++) {
        //this is needed to compute everything with respect to the center of the kernel.
        inputX = idx - nx / 2 + m;
        inputY = idy - ny / 2 + n;
        inputZ = idz - nz / 2 + l;

        if (inputX >= 0 && inputX < xsize && inputY >= 0 && inputY < ysize && inputZ >= 0 &&
            inputZ < zsize) {
          accumulation +=
              pow(image[(inputZ * xsize * ysize) + (inputX * ysize) + inputY] - mean, 2);
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

          accumulation +=
              pow(image[(inputZ * xsize * ysize) + (inputX * ysize) + inputY] - mean, 2);
        }
      }
    }
  }

  *standard_deviation = sqrt(accumulation / (nx * ny * nz));
}

static void get_gaussian_kernel_2d(double** kernel, int nx, int ny, float sigma) {
  /*

        kernel is given by the gaussian distribution:

        y = exp(||xi-xj||^2 /sigma^2)

    */

  //kernel allocation
  *kernel = (double*)malloc(sizeof(double) * nx * ny);

  if (!*kernel) {
    return;
  }

  int x;
  int y;

  int x0 = nx / 2;
  int y0 = ny / 2;

  double distance = 0;
  double normalization = 0;

  // Generate the kernel values.
  for (int i = 0; i < nx; i++) {

    for (int j = 0; j < ny; j++) {

      x = i - x0;
      y = j - y0;

      distance = x * x + y * y;

      (*kernel)[i * ny + j] = exp(-distance / (2 * sigma * sigma + 1E-16));
      normalization += (*kernel)[i * ny + j];

      //std::cout<<(*kernel)[i*ysize+j]<<" ";
    }

    //std::cout<<"\n";
  }

  for (int i = 0; i < nx * ny; i++) {
    (*kernel)[i] = (*kernel)[i] / normalization;
  }

}

static void get_gaussian_kernel_3d(double** kernel, int nx, int ny, int nz, float sigma) {
  /*

        kernel is given by the gaussian distribution:

        y = exp(||xi-xj||^2 /sigma^2)

    */

  //kernel allocation
  *kernel = (double*)malloc(sizeof(double) * nx * ny * nz);

  if (!*kernel) {
    return;
  }

  int x;
  int y;
  int z;

  int x0 = nx / 2;
  int y0 = ny / 2;
  int z0 = nz / 2;

  double distance = 0;
  double normalization = 0;

  // Generate the kernel values.
  for (int k = 0; k < nz; k++) {

    for (int i = 0; i < nx; i++) {

      for (int j = 0; j < ny; j++) {

        x = i - x0;
        y = j - y0;
        z = k - z0;

        distance = x * x + y * y + z * z;

        (*kernel)[k * nx * ny + i * ny + j] = exp(-distance / (2 * sigma * sigma + 1E-16));
        normalization += (*kernel)[k * nx * ny + i * ny + j];

        //std::cout<<(*kernel)[i*ny+j]<<" ";
      }

      //std::cout<<"\n";
    }
  }

  for (int i = 0; i < nx * ny * nz; i++) {
    (*kernel)[i] = (*kernel)[i] / normalization;
  }

}
#endif  // KERNELS_H