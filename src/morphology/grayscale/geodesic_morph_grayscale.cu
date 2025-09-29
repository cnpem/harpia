#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/geodesic_morph_grayscale.h"

/**
 * @brief Perform geodesic grayscale erosion/dilation operation for one pixel.
 *
 * This function applies geodesic morphological operations (erosion or dilation) 
 * to a single pixel in a 3D image. The function determines the new pixel value based on the 
 * neighborhood defined by the kernel and the given mask.
 *
 * @tparam dtype The data type of the image.
 * @param image Input image (marker image).
 * @param mask Mask image.
 * @param output Output image.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image.
 * @param padding_bottom Padding added at the bottom in the z-dimension.
 * @param padding_top Padding added at the top in the z-dimension.
 * @param centerIdx X-coordinate of the pixel being processed.
 * @param centerIdy Y-coordinate of the pixel being processed.
 * @param centerIdz Z-coordinate of the pixel being processed.
 * @param kernel_xsize Kernel size in the x-dimension.
 * @param kernel_ysize Kernel size in the y-dimension.
 * @param kernel_zsize Kernel size in the z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 * 
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.6, 
 *       on pages 667-668, adapted for grayscale images.
 * @see R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
 */
template <typename dtype>
CUDA_HOSTDEV void geodesic_morph_grayscale_pixel(dtype* image, dtype* mask, dtype* output,
                                                 const int xsize, const int ysize, const int zsize,
                                                 const int padding_bottom, const int padding_top,
                                                 int centerIdx, int centerIdy, int centerIdz,
                                                 int kernel_xsize, int kernel_ysize,
                                                 int kernel_zsize, MorphOp operation) {
  dtype* im = image;

  // Initialize auxiliary value with the central pixel
  size_t centerIndex = static_cast<size_t>(centerIdz) * ysize * xsize + 
                       static_cast<size_t>(centerIdy) * xsize + 
                       static_cast<size_t>(centerIdx);

  dtype aux = im[centerIndex];

  int startIdx = centerIdx - kernel_xsize / 2;
  int startIdy = centerIdy - kernel_ysize / 2;
  int startIdz = centerIdz - kernel_zsize / 2;

  size_t index;
  int imageIdx, imageIdy, imageIdz;

  // Erosion/dilation operation
  for (int iz = 0; iz < kernel_zsize; iz++) {
    for (int iy = 0; iy < kernel_ysize; iy++) {
      for (int ix = 0; ix < kernel_xsize; ix++) {

        imageIdx = startIdx + ix;
        imageIdy = startIdy + iy;
        imageIdz = startIdz + iz;

        // Ignore out of bounds pixels
        if (imageIdx < 0 || imageIdx > xsize - 1 || imageIdy < 0 || imageIdy > ysize - 1 ||
            imageIdz < -padding_bottom || imageIdz > zsize + padding_top - 1) {
        }

        else {
          index = static_cast<size_t>(imageIdz) * xsize * ysize + 
                  static_cast<size_t>(imageIdy) * xsize + 
                  static_cast<size_t>(imageIdx);

          if (operation == EROSION) {
            aux = (im[index] < aux) ? im[index] : aux;  // Erosion: aux is the min value
          } else {
            aux = (im[index] > aux) ? im[index] : aux;  // Dilation: aux is  the max value
          }
        }
      }
    }
  }

  // Point-wise maximun/minimun operation
  if (operation == EROSION) {
    output[centerIndex] =
        (aux > mask[centerIndex]) ? aux : mask[centerIndex];  // Erosion: output is the max value
  } else {
    output[centerIndex] =
        (aux < mask[centerIndex]) ? aux : mask[centerIndex];  // Dilation: output is the min value
  }
}
template CUDA_HOSTDEV void geodesic_morph_grayscale_pixel<int>(int*, int*, int*, const int,
                                                               const int, const int, const int,
                                                               const int, int, int, int, int, int,
                                                               int, MorphOp);
template CUDA_HOSTDEV void geodesic_morph_grayscale_pixel<unsigned int>(
    unsigned int*, unsigned int*, unsigned int*, const int, const int, const int, const int,
    const int, int, int, int, int, int, int, MorphOp);
template CUDA_HOSTDEV void geodesic_morph_grayscale_pixel<float>(float*, float*, float*, const int,
                                                                 const int, const int, const int,
                                                                 const int, int, int, int, int, int,
                                                                 int, MorphOp);
/**
 * @brief CUDA kernel for geodesic grayscale erosion/dilation on an entire image.
 *
 * @tparam dtype The data type of the image.
 * @param deviceImage Input image on the GPU.
 * @param deviceMask Mask image on the GPU.
 * @param deviceOutput Output image on the GPU.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image.
 * @param padding_bottom Padding at the bottom in the z-dimension.
 * @param padding_top Padding at the top in the z-dimension.
 * @param kernel_xsize Kernel size in x-dimension.
 * @param kernel_ysize Kernel size in y-dimension.
 * @param kernel_zsize Kernel size in z-dimension.
 * @param operation Morphological operation (EROSION or DILATION).
 * 
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.6, 
 *       on pages 667-668, adapted for grayscale images.
 * @see geodesic_morph_grayscale_pixel()
 */
template <typename dtype>
__global__ void geodesic_morph_grayscale_kernel(dtype* deviceImage, dtype* deviceMask,
                                                dtype* deviceOutput, const int xsize,
                                                const int ysize, const int zsize,
                                                const int padding_bottom, const int padding_top,
                                                int kernel_xsize, int kernel_ysize,
                                                int kernel_zsize, MorphOp operation) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < xsize && idy < ysize && idz < zsize) {
    geodesic_morph_grayscale_pixel(deviceImage, deviceMask, deviceOutput, xsize, ysize, zsize,
                                   padding_bottom, padding_top, idx, idy, idz, kernel_xsize,
                                   kernel_ysize, kernel_zsize, operation);
  }
}
template __global__ void geodesic_morph_grayscale_kernel<int>(int*, int*, int*, const int,
                                                              const int, const int, const int,
                                                              const int, int, int, int, MorphOp);
template __global__ void geodesic_morph_grayscale_kernel<unsigned int>(unsigned int*, unsigned int*,
                                                                       unsigned int*, const int,
                                                                       const int, const int,
                                                                       const int, const int, int,
                                                                       int, int, MorphOp);
template __global__ void geodesic_morph_grayscale_kernel<float>(float*, float*, float*, const int,
                                                                const int, const int, const int,
                                                                const int, int, int, int, MorphOp);

template <typename dtype>
void geodesic_morph_grayscale(dtype* deviceImage, dtype* deviceMask, dtype* deviceOutput,
                              const int xsize, const int ysize, const int zsize,
                              const int flag_verbose, const int padding_bottom,
                              const int padding_top, MorphOp operation) {

  //define connectivity kernel size for images of any dimension
  int kernel_xsize = (xsize > 2) ? 3 : xsize;
  int kernel_ysize = (ysize > 2) ? 3 : ysize;
  int kernel_zsize = (zsize > 2) ? 3 : zsize;

  //set up execution configuratio
  dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
  if (zsize == 1)
    block = dim3(BLOCK_2D, BLOCK_2D, 1);

  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  // check grid and block dimension from host side
  if (flag_verbose) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  // device erosion/dialation
  geodesic_morph_grayscale_kernel<<<grid, block>>>(
      deviceImage, deviceMask, deviceOutput, xsize, ysize, zsize, padding_bottom, padding_top,
      kernel_xsize, kernel_ysize, kernel_zsize, operation);
  cudaDeviceSynchronize();  //assures all gpu threads are fineshed
}
template void geodesic_morph_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                            const int, const int, const int, MorphOp);
template void geodesic_morph_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                     const int, const int, const int, const int,
                                                     const int, const int, MorphOp);
template void geodesic_morph_grayscale<float>(float*, float*, float*, const int, const int,
                                              const int, const int, const int, const int, MorphOp);

template <typename dtype>
void geodesic_morph_grayscale_on_device(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                        const int xsize, const int ysize, const int zsize,
                                        const int flag_verbose, const int padding_bottom,
                                        const int padding_top, MorphOp operation) {
  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);
  size_t nBytes_padding = xsize * ysize * (padding_bottom + padding_top) * sizeof(dtype);
  size_t nBytes_input = nBytes + nBytes_padding;

  // malloc device global memory
  dtype *deviceImage, *deviceOutput, *deviceMask, *i_deviceImage, *i_hostImage, *i_deviceMask,
      *i_hostMask;
  CHECK(cudaMalloc((dtype**)&i_deviceImage, nBytes_input));
  CHECK(cudaMalloc((dtype**)&i_deviceMask, nBytes_input));
  CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));

  // transfer input and mask + padding
  i_hostImage = hostImage - padding_bottom * xsize * ysize;
  i_hostMask = hostMask - padding_bottom * xsize * ysize;

  CHECK(cudaMemcpy(i_deviceImage, i_hostImage, nBytes_input, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(i_deviceMask, i_hostMask, nBytes_input, cudaMemcpyHostToDevice));

  deviceImage = i_deviceImage + padding_bottom * xsize * ysize;
  deviceMask = i_deviceMask + padding_bottom * xsize * ysize;

  // device erosion/dialation
  geodesic_morph_grayscale(deviceImage, deviceMask, deviceOutput, xsize, ysize, zsize, flag_verbose,
                           padding_bottom, padding_top, operation);

  // transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

  // free host memorys
  cudaFree(i_deviceImage);
  cudaFree(i_deviceMask);
  cudaFree(deviceOutput);
}
template void geodesic_morph_grayscale_on_device<int>(int*, int*, int*, const int, const int,
                                                      const int, const int, const int, const int,
                                                      MorphOp);
template void geodesic_morph_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*,
                                                               unsigned int*, const int, const int,
                                                               const int, const int, const int,
                                                               const int, MorphOp);
template void geodesic_morph_grayscale_on_device<float>(float*, float*, float*, const int,
                                                        const int, const int, const int, const int,
                                                        const int, MorphOp);

template <typename dtype>
void geodesic_morph_grayscale_on_host(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                      const int xsize, const int ysize, const int zsize,
                                      MorphOp operation) {

  //define connectivity kernel size for images of any dimension
  int kernel_xsize = (xsize > 2) ? 3 : xsize;
  int kernel_ysize = (ysize > 2) ? 3 : ysize;
  int kernel_zsize = (zsize > 2) ? 3 : zsize;

  for (int idz = 0; idz < zsize; idz++) {
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {

        geodesic_morph_grayscale_pixel(hostImage, hostMask, hostOutput, xsize, ysize, zsize, 0, 0,
                                       idx, idy, idz, kernel_xsize, kernel_ysize, kernel_zsize,
                                       operation);
      }
    }
  }  // slide over image
}
template void geodesic_morph_grayscale_on_host<int>(int*, int*, int*, const int, const int,
                                                    const int, MorphOp);
template void geodesic_morph_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*,
                                                             unsigned int*, const int, const int,
                                                             const int, MorphOp);
template void geodesic_morph_grayscale_on_host<float>(float*, float*, float*, const int, const int,
                                                      const int, MorphOp);
