#include <stdio.h>
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/morph_grayscale.h"
/**
 * @brief Perform a grayscale morphological operation (erosion or dilation) on a single pixel.
 *
 * This function applies a grayscale morphological operation (either erosion or dilation) to a specific pixel
 * in a 3D image, using a given kernel that defines the neighborhood for the operation.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param image Pointer to the input image.
 * @param output Pointer to the output image.
 * @param xsize Width of the image (number of pixels in the x-dimension).
 * @param ysize Height of the image (number of pixels in the y-dimension).
 * @param zsize Depth of the image (number of pixels in the z-dimension).
 * @param padding_bottom Padding size added to the bottom of the image.
 * @param padding_top Padding size added to the top of the image.
 * @param centerIdx x-coordinate of the center pixel where the operation is applied.
 * @param centerIdy y-coordinate of the center pixel where the operation is applied.
 * @param centerIdz z-coordinate of the center pixel where the operation is applied.
 * @param kernel Pointer to the kernel used for the morphological operation.
 * @param kernel_xsize Width of the kernel (number of elements in the x-dimension).
 * @param kernel_ysize Height of the kernel (number of elements in the y-dimension).
 * @param kernel_zsize Depth of the kernel (number of elements in the z-dimension).
 * @param operation The morphological operation to apply (EROSION or DILATION).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.8, 
 *       on pages 674-679.
 * @see R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
 */
template <typename dtype>
CUDA_HOSTDEV void morph_grayscale_pixel(dtype* image, dtype* output, const int xsize,
                                        const int ysize, const int zsize, const int padding_bottom,
                                        const int padding_top, int centerIdx, int centerIdy,
                                        int centerIdz, int* kernel, int kernel_xsize,
                                        int kernel_ysize, int kernel_zsize, MorphOp operation) {
  dtype* im = image;
  int* ik = kernel;

  // Initialize auxiliary value with the central pixel
  size_t centerPixelIndex = static_cast<size_t>(centerIdz) * xsize * ysize + 
                            static_cast<size_t>(centerIdy) * xsize + 
                            static_cast<size_t>(centerIdx);
  dtype aux = im[centerPixelIndex];

  size_t index;
  int imageIdx, imageIdy, imageIdz;

  int startIdx = centerIdx - kernel_xsize / 2;
  int startIdy = centerIdy - kernel_ysize / 2;
  int startIdz = centerIdz - kernel_zsize / 2;

  for (int iz = 0; iz < kernel_zsize; iz++) {
    for (int iy = 0; iy < kernel_ysize; iy++) {
      for (int ix = 0; ix < kernel_xsize; ix++) {

        imageIdx = startIdx + ix;
        imageIdy = startIdy + iy;
        imageIdz = startIdz + iz;

        // Ignore out of bounds pixels and don't care pixels
        if (imageIdx < 0 || imageIdx > xsize - 1 || imageIdy < 0 || imageIdy > ysize - 1 ||
            imageIdz < -padding_bottom || imageIdz > zsize + padding_top - 1 || ik[ix] < 0) {
          // do nothing.
        }

        else {
          index = static_cast<size_t>(imageIdz) * xsize * ysize + 
                  static_cast<size_t>(imageIdy) * xsize + 
                  static_cast<size_t>(imageIdx);
          
          if (operation == EROSION) {
            aux = (im[index] < aux) ? im[index] : aux;  // Erosion: aux is the min value
          } else {
            aux = (im[index] > aux) ? im[index] : aux;  // Dilation: aux is the max value
          }
        }
      }
    }
  }
  output[centerPixelIndex] = aux;
}
template CUDA_HOSTDEV void morph_grayscale_pixel<unsigned int>(unsigned int*, unsigned int*,
                                                               const int, const int, const int,
                                                               const int, const int, int, int, int,
                                                               int*, int, int, int,

                                                               MorphOp);
template CUDA_HOSTDEV void morph_grayscale_pixel<int>(int*, int*, const int, const int, const int,
                                                      const int, const int, int, int, int, int*,
                                                      int, int, int, MorphOp);
template CUDA_HOSTDEV void morph_grayscale_pixel<float>(float*, float*, const int, const int,
                                                        const int, const int, const int, int, int,
                                                        int, int*, int, int, int, MorphOp);
/**
 * @brief CUDA kernel to perform a grayscale morphological operation on a 3D image.
 *
 * This kernel function is executed on the GPU, applying a morphological grayscale operation (erosion or dilation)
 * to every pixel in the image. Each thread processes a single pixel by invoking `morph_grayscale_pixel`
 * for the corresponding pixel.
 *
 * @tparam dtype The data type of the image (e.g., int, unsigned int, uint16_t, etc.).
 * @param deviceImage Pointer to the input image stored in GPU memory.
 * @param deviceOutput Pointer to the output image stored in GPU memory.
 * @param xsize Width of the image (number of pixels in the x-dimension).
 * @param ysize Height of the image (number of pixels in the y-dimension).
 * @param zsize Depth of the image (number of pixels in the z-dimension).
 * @param padding_bottom Padding size added to the bottom of the image.
 * @param padding_top Padding size added to the top of the image.
 * @param kernel Pointer to the kernel used for the morphological operation.
 * @param kernel_xsize Width of the kernel (number of elements in the x-dimension).
 * @param kernel_ysize Height of the kernel (number of elements in the y-dimension).
 * @param kernel_zsize Depth of the kernel (number of elements in the z-dimension).
 * @param operation The morphological operation to apply (EROSION or DILATION).
 *
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.8, 
 *       on pages 674-679.
 * @see R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
 */
template <typename dtype>
__global__ void morph_grayscale_kernel(dtype* deviceImage, dtype* deviceOutput, const int xsize,
                                       const int ysize, const int zsize, const int padding_bottom,
                                       const int padding_top, int* kernel, int kernel_xsize,
                                       int kernel_ysize, int kernel_zsize, MorphOp operation) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    morph_grayscale_pixel(deviceImage, deviceOutput, xsize, ysize, zsize, padding_bottom,
                          padding_top, idx, idy, idz, kernel, kernel_xsize, kernel_ysize,
                          kernel_zsize, operation);
  }
}
template __global__ void morph_grayscale_kernel<unsigned int>(unsigned int*, unsigned int*,
                                                              const int, const int, const int,
                                                              const int, const int, int*, int, int,
                                                              int, MorphOp);
template __global__ void morph_grayscale_kernel<int>(int*, int*, const int, const int, const int,
                                                     const int, const int, int*, int, int, int,
                                                     MorphOp);
template __global__ void morph_grayscale_kernel<float>(float*, float*, const int, const int,
                                                       const int, const int, const int, int*, int,
                                                       int, int, MorphOp);

template <typename dtype>
void morph_grayscale(dtype* deviceImage, dtype* deviceOutput, const int xsize, const int ysize,
                     const int zsize, const int flag_verbose, const int padding_bottom,
                     const int padding_top, int* deviceKernel, int kernel_xsize, int kernel_ysize,
                     int kernel_zsize, MorphOp operation) {
  // Set up execution configuration
  dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
  if (zsize == 1)
    block = dim3(BLOCK_2D, BLOCK_2D, 1);
  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  // Check grid and block dimensions from host side
  if (flag_verbose) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  // Device erosion/dilation
  morph_grayscale_kernel<<<grid, block>>>(deviceImage, deviceOutput, xsize, ysize, zsize,
                                          padding_bottom, padding_top, deviceKernel, kernel_xsize,
                                          kernel_ysize, kernel_zsize, operation);
  cudaDeviceSynchronize();  // Assures all GPU threads are finished
}
template void morph_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                            const int, const int, const int, const int, int*, int,
                                            int, int, MorphOp);
template void morph_grayscale<int>(int*, int*, const int, const int, const int, const int,
                                   const int, const int, int*, int, int, int, MorphOp);
template void morph_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                     const int, const int, int*, int, int, int, MorphOp);

template <typename dtype>
void morph_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, const int flag_verbose,
                               const int padding_bottom, const int padding_top, int* kernel,
                               int kernel_xsize, int kernel_ysize, int kernel_zsize,
                               MorphOp operation) {
  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);
  size_t nBytes_padding = xsize * ysize * (padding_bottom + padding_top) * sizeof(dtype);
  size_t nBytes_input = nBytes + nBytes_padding;

  // set kenrel dimension
  size_t kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
  size_t kernel_nBytes = kernel_size * sizeof(int);

  // malloc device global memory
  dtype *deviceImage, *deviceOutput, *i_deviceImage, *i_hostImage;
  int* deviceKernel;
  CHECK(cudaMalloc((dtype**)&i_deviceImage, nBytes_input));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
  CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

  // transfer data from the host to the device
  CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

  // transfer input + padding
  i_hostImage = hostImage - padding_bottom * xsize * ysize;

  CHECK(cudaMemcpy(i_deviceImage, i_hostImage, nBytes_input, cudaMemcpyHostToDevice));

  deviceImage = i_deviceImage + padding_bottom * xsize * ysize;

  // device erosion/dialation

  morph_grayscale(deviceImage, deviceOutput, xsize, ysize, zsize, flag_verbose, padding_bottom,
                  padding_top, deviceKernel, kernel_xsize, kernel_ysize, kernel_zsize, operation);

  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free host memorys

  cudaFree(i_deviceImage);  
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);
}
template void morph_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                      const int, const int, const int, const int,
                                                      const int, int*, int, int, int, MorphOp);
template void morph_grayscale_on_device<int>(int*, int*, const int, const int, const int, const int,
                                             const int, const int, int*, int, int, int, MorphOp);
template void morph_grayscale_on_device<float>(float*, float*, const int, const int, const int,
                                               const int, const int, const int, int*, int, int, int,
                                               MorphOp);

template <typename dtype>
void morph_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                             const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
                             int kernel_zsize, MorphOp operation) {
  for (int idz = 0; idz < zsize; idz++) {
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {

        morph_grayscale_pixel(hostImage, hostOutput, xsize, ysize, zsize, 0, 0, idx, idy, idz,
                              kernel, kernel_xsize, kernel_ysize, kernel_zsize, operation);
      }
    }
  }  // Slide over image
}
template void morph_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                    const int, const int, int*, int, int, int,
                                                    MorphOp);
template void morph_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*, int,
                                           int, int, MorphOp);
template void morph_grayscale_on_host<float>(float*, float*, const int, const int, const int, int*,
                                             int, int, int, MorphOp);