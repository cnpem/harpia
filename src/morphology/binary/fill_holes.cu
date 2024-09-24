// #include <stdio.h>
// #include <cstdint>  // For uint16_t, unsigned int
// #include "../../../include/common/grid_block_sizes.h"
// #include "../../../include/morphology/complement_binary.h"
// #include "../../../include/morphology/cuda_helper.h"
// #include "../../../include/morphology/fill_holes.h"
// #include "../../../include/morphology/morphology.h"
// #include "../../../include/morphology/reconstruction_binary.h"

// template <typename dtype>
// __global__ void fill_holes_marker(dtype* deviceImage, dtype* deviceOutput, const int xsize,
//                                   const int ysize, const int zsize) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   int idy = threadIdx.y + blockIdx.y * blockDim.y;
//   int idz = threadIdx.z + blockIdx.z * blockDim.z;

//   //define connectivity kernel size for images of any dimension
//   //when one of the dimensions is of size 1,the hole image is a border for that dimention
//   //this flags avoid this behavier
//   bool ignore_xsize = (xsize == 1) ? true : false;
//   bool ignore_ysize = (ysize == 1) ? true : false;
//   bool ignore_zsize = (zsize == 1) ? true : false;

//   if (idx < xsize && idy < ysize && idz < zsize) {
//     int index = idz * xsize * ysize + idy * xsize + idx;

//     // Check if the current voxel is on the border
//     bool is_border_x = (idx == 0 || idx == xsize - 1) && (!ignore_xsize);
//     bool is_border_y = (idy == 0 || idy == ysize - 1) && (!ignore_ysize);
//     bool is_border_z = (idz == 0 || idz == zsize - 1) && (!ignore_zsize);
//     bool is_border = is_border_x || is_border_y || is_border_z;

//     if (is_border) {
//       deviceOutput[index] = 1 - deviceImage[index];  // Set border to complement
//     } else {
//       deviceOutput[index] = 0;  // Set inner pixels to zero
//     }
//   }
// }
// template __global__ void fill_holes_marker<unsigned int>(unsigned int*, unsigned int*, const int,
//                                                          const int, const int);
// template __global__ void fill_holes_marker<int>(int*, int*, const int, const int, const int);
// template __global__ void fill_holes_marker<uint16_t>(uint16_t*, uint16_t*, const int, const int,
//                                                      const int);

// /**
//  * @brief Performs the bottom-hat transformation on the input image on the device (GPU).
//  *
//  * @tparam dtype Data type of the image.
//  * @param hostImage Pointer to the input image on the host.
//  * @param hostOutput Pointer to the output image on the host.
//  * @param xsize Size of the image in the x-dimension.
//  * @param ysize Size of the image in the y-dimension.
//  * @param zsize Size of the image in the z-dimension.
//  * @param flag_verbose Flag for verbose output.
//  */
// template <typename dtype>
// void fill_holes_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                           const int zsize, const int flag_verbose) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Malloc device global memory
//   dtype *deviceMarker, *deviceMask, *deviceAux;
//   CHECK(cudaMalloc((dtype**)&deviceAux, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceMarker, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceMask, nBytes));

//   // Transfer data from the host to the device
//   CHECK(cudaMemcpy(deviceAux, hostImage, nBytes, cudaMemcpyHostToDevice));
//   CHECK(cudaMemset(deviceMarker, 0, nBytes));  //Initialize with zeros
//   CHECK(cudaMemset(deviceMask, 0, nBytes));

//   // Prepare marker
//   dim3 block(BLOCK_3D, BLOCK_3D, BLOCK_3D);
//   if (zsize == 1)
//     block = dim3(BLOCK_2D, BLOCK_2D, 1);
//   dim3 grid((xsize + block.x - 1) / block.x, (ysize + block.y - 1) / block.y,
//             (zsize + block.z - 1) / block.z);

//   if (flag_verbose) {
//     printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
//     printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
//   }

//   fill_holes_marker<<<grid, block>>>(deviceAux, deviceMarker, xsize, ysize, zsize);
//   cudaDeviceSynchronize();  // Assures all GPU threads are finished

//   // Prepare mask
//   complement_binary(deviceAux, deviceMask, size, flag_verbose);

//   // Reconstruction + Complement
//   reconstruction_binary(deviceMarker, deviceAux, xsize, ysize, zsize, deviceMask, DILATION,
//                         flag_verbose);
//   complement_binary(deviceAux, deviceAux, size, flag_verbose);

//   // Transfer data from the device to the host
//   CHECK(cudaMemcpy(hostOutput, deviceAux, nBytes, cudaMemcpyDeviceToHost));

//   // Free device memory
//   cudaFree(deviceAux);
//   cudaFree(deviceMarker);
//   cudaFree(deviceMask);
// }
// // Template instantiations for specific types
// template void fill_holes_on_device<int>(int*, int*, const int, const int, const int, const int);
// template void fill_holes_on_device<unsigned int>(unsigned int*, unsigned int*, const int, const int,
//                                                  const int, const int);
// template void fill_holes_on_device<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
//                                              const int);

// /**
//  * @brief Performs the bottom-hat transformation on the input image on the host (CPU).
//  *
//  * @tparam dtype Data type of the image.
//  * @param hostImage Pointer to the input image on the host.
//  * @param hostOutput Pointer to the output image on the host.
//  * @param xsize Size of the image in the x-dimension.
//  * @param ysize Size of the image in the y-dimension.
//  * @param zsize Size of the image in the z-dimension.
//  * @param flag_verbose Flag for verbose output.
//  */
// template <typename dtype>
// void fill_holes_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                         const int zsize) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Malloc device global memory
//   dtype* hostMarker = (dtype*)malloc(nBytes);
//   dtype* hostMask = (dtype*)malloc(nBytes);

//   //define connectivity kernel size for images of any dimension
//   //when one of the dimensions is of size 1,the hole image is a border for that dimention
//   //this flags avoid this behavier
//   bool ignore_xsize = (xsize == 1) ? true : false;
//   bool ignore_ysize = (ysize == 1) ? true : false;
//   bool ignore_zsize = (zsize == 1) ? true : false;

//   // Prepare marker
//   for (int idz = 0; idz < zsize; idz++) {
//     for (int idy = 0; idy < ysize; idy++) {
//       for (int idx = 0; idx < xsize; idx++) {
//         int index = idz * xsize * ysize + idy * xsize + idx;
//         // Check if the current voxel is on the border
//         bool is_border_x = (idx == 0 || idx == xsize - 1) && (!ignore_xsize);
//         bool is_border_y = (idy == 0 || idy == ysize - 1) && (!ignore_ysize);
//         bool is_border_z = (idz == 0 || idz == zsize - 1) && (!ignore_zsize);
//         bool is_border = is_border_x || is_border_y || is_border_z;

//         if (is_border) {
//           hostMarker[index] = 1 - hostImage[index];  // Set border to complement
//         } else {
//           hostMarker[index] = 0;  // Set inner pixels to zero
//         }
//       }
//     }
//   }  // slide over image

//   // // Prepare mask
//   complement_binary_on_host(hostImage, hostMask, size);

//   // // Reconstruction + Complement
//   reconstruction_binary_on_host(hostMarker, hostOutput, xsize, ysize, zsize, hostMask, DILATION);
//   complement_binary_on_host(hostOutput, hostOutput, size);
//   free(hostMarker);
//   free(hostMask);
// }
// // Template instantiations for specific types
// template void fill_holes_on_host<int>(int*, int*, const int, const int, const int);
// template void fill_holes_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
//                                                const int);
// template void fill_holes_on_host<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int);