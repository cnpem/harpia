// #include <stdio.h>
// #include "../../../include/common/grid_block_sizes.h"
// #include "../../../include/morphology/cuda_helper.h"
// #include "../../../include/morphology/morph_chain_grayscale.h"
// #include "../../../include/morphology/morph_grayscale.h"
// #include "../../../include/morphology/morphology.h"
// #include "../../../include/morphology/subtraction.h"

// /**
//  * @brief Performs the bottom-hat transformation on the input image on the device (GPU).
//  *
//  * @tparam dtype Data type of the image.
//  * @param hostImage Pointer to the input image on the host.
//  * @param hostOutput Pointer to the output image on the host.
//  * @param kernel Pointer to the morphological kernel.
//  * @param kernel_xsize Size of the kernel in the x-dimension.
//  * @param kernel_ysize Size of the kernel in the y-dimension.
//  * @param kernel_zsize Size of the kernel in the z-dimension.
//  * @param xsize Size of the image in the x-dimension.
//  * @param ysize Size of the image in the y-dimension.
//  * @param zsize Size of the image in the z-dimension.
//  * @param flag_verbose Flag for verbose output.
//  */
// template <typename dtype>
// void bottom_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                           const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
//                           int kernel_zsize, const int flag_verbose) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Set kernel dimension
//   int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
//   size_t kernel_nBytes = kernel_size * sizeof(int);

//   // Malloc device global memory
//   dtype *deviceImage, *deviceTmp, *deviceOutput;
//   int* deviceKernel;
//   CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceOutput, nBytes));
//   CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

//   // Transfer data from the host to the device
//   CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
//   CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

//   // Closing operation
//   morph_grayscale(deviceImage, deviceTmp, xsize, ysize, zsize, deviceKernel, kernel_xsize,
//                   kernel_ysize, kernel_zsize, DILATION, flag_verbose);
//   morph_grayscale(deviceTmp, deviceOutput, xsize, ysize, zsize, deviceKernel, kernel_xsize,
//                   kernel_ysize, kernel_zsize, EROSION, flag_verbose);
//   // B_hat = closing - f
//   subtraction(deviceImage, deviceOutput, xsize * ysize * zsize, flag_verbose);

//   // Transfer data from the device to the host
//   CHECK(cudaMemcpy(hostOutput, deviceOutput, nBytes, cudaMemcpyDeviceToHost));

//   // Free device memory
//   cudaFree(deviceTmp);
//   cudaFree(deviceImage);
//   cudaFree(deviceOutput);
//   cudaFree(deviceKernel);
// }
// // Template instantiations for specific types
// template void bottom_hat_on_device<int>(int*, int*, const int, const int, const int, int*, int, int,
//                                         int, const int);
// template void bottom_hat_on_device<unsigned int>(unsigned int*, unsigned int*, const int, const int,
//                                                  const int, int*, int, int, int, const int);
// template void bottom_hat_on_device<float>(float*, float*, const int, const int, const int, int*,
//                                           int, int, int, const int);

// /**
//  * @brief Performs the bottom-hat transformation on the input image on the host (CPU).
//  *
//  * @tparam dtype Data type of the image.
//  * @param hostImage Pointer to the input image on the host.
//  * @param hostOutput Pointer to the output image on the host.
//  * @param kernel Pointer to the morphological kernel.
//  * @param kernel_xsize Size of the kernel in the x-dimension.
//  * @param kernel_ysize Size of the kernel in the y-dimension.
//  * @param kernel_zsize Size of the kernel in the z-dimension.
//  * @param xsize Size of the image in the x-dimension.
//  * @param ysize Size of the image in the y-dimension.
//  * @param zsize Size of the image in the z-dimension.
//  * @param flag_verbose Flag for verbose output.
//  */
// template <typename dtype>
// void bottom_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                         const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
//                         int kernel_zsize) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;

//   // Opening operation
//   MorphChain closing = {DILATION, EROSION};
//   morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
//                                 kernel_ysize, kernel_zsize, closing);

//   // B_hat = closing - f
//   subtraction_on_host(hostImage, hostOutput, size);
// }
// // Template instantiations for specific types
// template void bottom_hat_on_host<int>(int*, int*, const int, const int, const int, int*, int, int,
//                                       int);
// template void bottom_hat_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
//                                                const int, int*, int, int, int);
// template void bottom_hat_on_host<float>(float*, float*, const int, const int, const int, int*, int,
//                                         int, int);