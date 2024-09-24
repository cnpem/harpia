// #include <stdio.h>
// #include "../../../include/common/grid_block_sizes.h"
// #include "../../../include/morphology/cuda_helper.h"
// #include "../../../include/morphology/morph_chain_grayscale.h"
// #include "../../../include/morphology/morph_grayscale.h"
// #include "../../../include/morphology/morphology.h"
// #include "../../../include/morphology/reconstruction_grayscale.h"
// #include "../../../include/morphology/subtraction.h"

// /**
//  * @brief Perform top-hat operation on the device.
//  *
//  * @tparam dtype Data type of the image.
//  * @param hostImage Pointer to the input image on the host.
//  * @param hostOutput Pointer to the output image on the host.
//  * @param kernel Pointer to the structuring element.
//  * @param kernel_xsize Size of the structuring element in the x-dimension.
//  * @param kernel_ysize Size of the structuring element in the y-dimension.
//  * @param kernel_zsize Size of the structuring element in the z-dimension.
//  * @param xsize Size of the image in the x-dimension.
//  * @param ysize Size of the image in the y-dimension.
//  * @param zsize Size of the image in the z-dimension.
//  * @param flag_verbose Flag for verbose output.
//  */
// template <typename dtype>
// void top_hat_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                        const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
//                        int kernel_zsize, const int flag_verbose) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Set kernel dimension
//   int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
//   size_t kernel_nBytes = kernel_size * sizeof(int);

//   // Malloc device global memory
//   dtype *deviceImage, *deviceTmp, *deviceAux;
//   int* deviceKernel;
//   CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceAux, nBytes));
//   CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

//   // Transfer data from the host to the device
//   CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
//   CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

//   // Opening operation: erosion followed by dilation
//   morph_grayscale(deviceImage, deviceAux, xsize, ysize, zsize, deviceKernel, kernel_xsize,
//                   kernel_ysize, kernel_zsize, EROSION, flag_verbose);
//   morph_grayscale(deviceAux, deviceTmp, xsize, ysize, zsize, deviceKernel, kernel_xsize,
//                   kernel_ysize, kernel_zsize, DILATION, flag_verbose);

//   // Top-hat: input - opening
//   subtraction(deviceTmp, deviceImage, size, flag_verbose);

//   // Transfer data from the device to the host
//   CHECK(cudaMemcpy(hostOutput, deviceImage, nBytes, cudaMemcpyDeviceToHost));

//   // Free device memory
//   cudaFree(deviceTmp);
//   cudaFree(deviceImage);
//   cudaFree(deviceAux);
//   cudaFree(deviceKernel);
// }
// // Template instantiations for specific types
// template void top_hat_on_device<int>(int*, int*, const int, const int, const int, int*, int, int,
//                                      int, const int);
// template void top_hat_on_device<unsigned int>(unsigned int*, unsigned int*, const int, const int,
//                                               const int, int*, int, int, int, const int);
// template void top_hat_on_device<float>(float*, float*, const int, const int, const int, int*, int,
//                                        int, int, const int);

// /**
//  * @brief Perform top-hat operation on the host.
//  *
//  * @tparam dtype Data type of the image.
//  * @param hostImage Pointer to the input image on the host.
//  * @param hostOutput Pointer to the output image on the host.
//  * @param kernel Pointer to the structuring element.
//  * @param kernel_xsize Size of the structuring element in the x-dimension.
//  * @param kernel_ysize Size of the structuring element in the y-dimension.
//  * @param kernel_zsize Size of the structuring element in the z-dimension.
//  * @param xsize Size of the image in the x-dimension.
//  * @param ysize Size of the image in the y-dimension.
//  * @param zsize Size of the image in the z-dimension.
//  * @param flag_verbose Flag for verbose output.
//  */
// template <typename dtype>
// void top_hat_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                      const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
//                      int kernel_zsize) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Allocate temporary memory
//   dtype* host_tmp = (dtype*)malloc(nBytes);

//   // Set input data
//   memset(host_tmp, 0, nBytes);

//   // Opening operation
//   MorphChain opening = {EROSION, DILATION};
//   morph_chain_grayscale_on_host(hostImage, host_tmp, xsize, ysize, zsize, kernel, kernel_xsize,
//                                 kernel_ysize, kernel_zsize, opening);

//   // Em vez de fazer apenas um openning, usa o opening simples como marker pra um opening by
//   // reconstruction
//   // TODO: substituir a subtraçõa por uma reconstrução geodésica p/ grayscale usando a imagem de
//   // 'opening' como 'marker', depois da convergênica da reconstrução, é feita a subtração!!

//   // Top-hat: f - opening
//   memcpy(hostOutput, hostImage, nBytes);
//   subtraction_on_host(host_tmp, hostOutput, size);

//   // Free temporary memory
//   free(host_tmp);
// }
// // Template instantiations for specific types
// template void top_hat_on_host<int>(int*, int*, const int, const int, const int, int*, int, int,
//                                    int);
// template void top_hat_on_host<unsigned int>(unsigned int*, unsigned int*, const int, const int,
//                                             const int, int*, int, int, int);
// template void top_hat_on_host<float>(float*, float*, const int, const int, const int, int*, int,
//                                      int, int);

// //##################################################################################################
// // AVISO VERSION -----------------------------------------------------------------------------------
// //##################################################################################################
// // Reference: https://www.thermofisher.com/software-em-3d-vis/xtra-library/xtras/interactive-top-hat-by-reconstruction
// template <typename dtype>
// void top_hat_aviso_on_device(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                              const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
//                              int kernel_zsize, const int flag_verbose) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Set kernel dimension
//   int kernel_size = kernel_xsize * kernel_ysize * kernel_zsize;
//   size_t kernel_nBytes = kernel_size * sizeof(int);

//   // Malloc device global memory
//   dtype *deviceImage, *deviceTmp, *deviceAux;
//   int* deviceKernel;
//   CHECK(cudaMalloc((dtype**)&deviceImage, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceTmp, nBytes));
//   CHECK(cudaMalloc((dtype**)&deviceAux, nBytes));
//   CHECK(cudaMalloc((int**)&deviceKernel, kernel_nBytes));

//   // Transfer data from the host to the device
//   CHECK(cudaMemcpy(deviceImage, hostImage, nBytes, cudaMemcpyHostToDevice));
//   CHECK(cudaMemcpy(deviceKernel, kernel, kernel_nBytes, cudaMemcpyHostToDevice));

//   // Opening operation: erosion followed by dilation
//   morph_grayscale(deviceImage, deviceAux, xsize, ysize, zsize, deviceKernel, kernel_xsize,
//                   kernel_ysize, kernel_zsize, EROSION, flag_verbose);
//   morph_grayscale(deviceAux, deviceTmp, xsize, ysize, zsize, deviceKernel, kernel_xsize,
//                   kernel_ysize, kernel_zsize, DILATION, flag_verbose);

//   reconstruction_grayscale(deviceTmp, deviceAux, xsize, ysize, zsize, deviceImage, DILATION,
//                            flag_verbose);

//   // Top-hat: input - opening
//   subtraction(deviceAux, deviceImage, size, flag_verbose);

//   // Transfer data from the device to the host
//   CHECK(cudaMemcpy(hostOutput, deviceImage, nBytes, cudaMemcpyDeviceToHost));

//   // Free device memory
//   cudaFree(deviceTmp);
//   cudaFree(deviceImage);
//   cudaFree(deviceAux);
//   cudaFree(deviceKernel);
// }
// // Template instantiations for specific types
// template void top_hat_aviso_on_device<int>(int*, int*, const int, const int, const int, int*, int,
//                                            int, int, const int);
// template void top_hat_aviso_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
//                                                     const int, const int, int*, int, int, int,
//                                                     const int);
// template void top_hat_aviso_on_device<float>(float*, float*, const int, const int, const int, int*,
//                                              int, int, int, const int);

// template <typename dtype>
// void top_hat_aviso_on_host(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
//                            const int zsize, int* kernel, int kernel_xsize, int kernel_ysize,
//                            int kernel_zsize) {
//   // Set input dimension
//   int size = xsize * ysize * zsize;
//   size_t nBytes = size * sizeof(dtype);

//   // Allocate temporary memory
//   dtype* host_tmp = (dtype*)malloc(nBytes);

//   // Set input data
//   memset(host_tmp, 0, nBytes);

//   // Opening operation
//   MorphChain opening = {EROSION, DILATION};
//   morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
//                                 kernel_ysize, kernel_zsize, opening);

//   // Em vez de fazer apenas um openning, usa o opening simples como marker pra um opening by
//   // reconstruction
//   // TODO: substituir a subtraçõa por uma reconstrução geodésica p/ grayscale usando a imagem de
//   // 'opening' como 'marker', depois da convergênica da reconstrução, é feita a subtração!!

//   reconstruction_grayscale_on_host(hostOutput, host_tmp, xsize, ysize, zsize, hostImage, DILATION);

//   // Top-hat: f - opening
//   memcpy(hostOutput, hostImage, nBytes);
//   subtraction_on_host(host_tmp, hostOutput, size);

//   // Free temporary memory
//   free(host_tmp);
// }
// // Template instantiations for specific types
// template void top_hat_aviso_on_host<int>(int*, int*, const int, const int, const int, int*, int,
//                                          int, int);
// template void top_hat_aviso_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
//                                                   const int, const int, int*, int, int, int);
// template void top_hat_aviso_on_host<float>(float*, float*, const int, const int, const int, int*,
//                                            int, int, int);