#include <stdio.h>
#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/grid_block_sizes.h"
#include "../../../include/morphology/complement_binary.h"
#include "../../../include/morphology/cuda_helper.h"
#include "../../../include/morphology/fill_holes.h"
#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/reconstruction_binary.h"

/**
 * @brief Identifies and marks the holes in a 3D image using a connectivity-based approach.
 *
 * This kernel processes each voxel in the input image and marks the border regions based on 
 * connectivity. The output marker image is used as a seed for further processing, such as 
 * morphological reconstruction.
 *
 * @tparam dtype Data type of the image.
 * @param deviceImage Pointer to the input image on the GPU.
 * @param deviceOutput Pointer to the output marker image on the GPU.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * 
 * @note This implementation is based on the morphological operations 
 *       described in "Digital Image Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, 
 *       particularly in Chapter 9 (Morphological Image Processing), Section 9.6, 
 *       on pages 671-672.
 * @see R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
 */ 
template <typename dtype>
__global__ void fill_holes_marker(dtype* deviceImage, dtype* deviceOutput, const int xsize,
                                  const int ysize, const int zsize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  int idz = threadIdx.z + blockIdx.z * blockDim.z;

  // Define connectivity kernel size for images of any dimension
  // Avoid treating a flat image as a border by using these flags
  bool ignore_xsize = (xsize == 1) ? true : false;
  bool ignore_ysize = (ysize == 1) ? true : false;
  bool ignore_zsize = (zsize == 1) ? true : false;

  if (idx < xsize && idy < ysize && idz < zsize) {
    size_t index = idz * xsize * ysize + idy * xsize + idx;

    // Check if the current voxel is on the border
    bool is_border_x = (idx == 0 || idx == xsize - 1) && (!ignore_xsize);
    bool is_border_y = (idy == 0 || idy == ysize - 1) && (!ignore_ysize);
    bool is_border_z = (idz == 0 || idz == zsize - 1) && (!ignore_zsize);
    bool is_border = is_border_x || is_border_y || is_border_z;

    if (is_border) {
      deviceOutput[index] = 1 - deviceImage[index];  // Set border to complement
    } else {
      deviceOutput[index] = 0;  // Set inner pixels to zero
    }
  }
}
template __global__ void fill_holes_marker<unsigned int>(unsigned int*, unsigned int*, const int,
                                                         const int, const int);
template __global__ void fill_holes_marker<int>(int*, int*, const int, const int, const int);
template __global__ void fill_holes_marker<int16_t>(int16_t*, int16_t*, const int, const int,
                                                    const int);
template __global__ void fill_holes_marker<uint16_t>(uint16_t*, uint16_t*, const int, const int,
                                                     const int);

template <typename dtype>
void fill_holes_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, const int flag_verbose) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Allocate device memory
  dtype *deviceMarker, *deviceMask, *deviceAux;
  CHECK(cudaMalloc((dtype**)&deviceAux, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceMarker, nBytes));
  CHECK(cudaMalloc((dtype**)&deviceMask, nBytes));

  // Transfer data from the host to the device
  CHECK(cudaMemcpy(deviceAux, hostImage, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(deviceMarker, 0, nBytes));  //Initialize with zeros
  CHECK(cudaMemset(deviceMask, 0, nBytes));

  // Configure execution parameters
  dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
  if (zsize == 1)
    block = dim3(BLOCK_2D, BLOCK_2D, 1);
  dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
            (zsize + block.z - 1) / block.z);

  if (flag_verbose) {
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
  }

  // Create marker for morphological reconstruction
  fill_holes_marker<<<grid, block>>>(deviceAux, deviceMarker, xsize, ysize, zsize);
  cudaDeviceSynchronize();  // Assures all GPU threads are finished

  // Prepare mask for morphological reconstruction
  complement_binary(deviceAux, deviceMask, size, flag_verbose);

  // Perform morphological reconstruction (hole filling)
  reconstruction_binary(deviceMarker, deviceMask, deviceAux, xsize, ysize, zsize, DILATION,
                        flag_verbose);
  complement_binary(deviceAux, deviceAux, size, flag_verbose);

  // Transfer data from the device to the host
  CHECK(cudaMemcpy(hostOutput, deviceAux, nBytes, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(deviceAux);
  cudaFree(deviceMarker);
  cudaFree(deviceMask);
}
template void fill_holes_on_device<int>(int*, int*, const int, const int, const int, const int);
template void fill_holes_on_device<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                                 const int, const int);
template void fill_holes_on_device<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                            const int);
template void fill_holes_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                             const int);

template <typename dtype>
void fill_holes_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize) {
  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(dtype);

  // Malloc device global memory
  dtype* hostMarker = (dtype*)malloc(nBytes);
  dtype* hostMask = (dtype*)malloc(nBytes);

  // Define connectivity kernel size for images of any dimension
  // Avoid treating a flat image as a border by using these flags
  bool ignore_xsize = (xsize == 1) ? true : false;
  bool ignore_ysize = (ysize == 1) ? true : false;
  bool ignore_zsize = (zsize == 1) ? true : false;

  // Prepare marker
  for (int idz = 0; idz < zsize; idz++) {
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {
        size_t index = static_cast<size_t>(idz) * xsize * ysize + 
                       static_cast<size_t>(idy) * xsize + 
                       static_cast<size_t>(idx);
        // Check if the current voxel is on the border
        bool is_border_x = (idx == 0 || idx == xsize - 1) && (!ignore_xsize);
        bool is_border_y = (idy == 0 || idy == ysize - 1) && (!ignore_ysize);
        bool is_border_z = (idz == 0 || idz == zsize - 1) && (!ignore_zsize);
        bool is_border = is_border_x || is_border_y || is_border_z;

        if (is_border) {
          hostMarker[index] = 1 - hostImage[index];  // Set border to complement
        } else {
          hostMarker[index] = 0;  // Set inner pixels to zero
        }
      }
    }
  }  // slide over image

  // Prepare mask
  complement_binary_on_host(hostImage, hostMask, size);

  // Reconstruction + Complement
  reconstruction_binary_on_host(hostMarker, hostMask, hostOutput, xsize, ysize, zsize, DILATION);
  complement_binary_on_host(hostOutput, hostOutput, size);
  free(hostMarker);
  free(hostMask);
}
template void fill_holes_on_host<int>(int*, int*, const int, const int, const int);
template void fill_holes_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                               const int);
template void fill_holes_on_host<int16_t>(int16_t*, int16_t*, const int, const int, const int);
template void fill_holes_on_host<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int);