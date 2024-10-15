#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/bottom_hat.h"
#include "../../../include/morphology/geodesic_morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/operations_grayscale.h"
#include "../../../include/morphology/reconstruction_grayscale.h"
#include "../../../include/morphology/top_hat.h"
#include "../../../include/morphology/top_hat_reconstruction.h"

/**
 * @brief Perform erosion operation on a grayscale image.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */

template <typename dtype>
void erosion_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {
  if (gpu) {
    int ncopies = 2;
    int flag_chain = 0;
    chunkedExecutorKernel(morph_grayscale_on_device<dtype>, ncopies, gpuMemory, flag_chain,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, EROSION);
  }

  else {
    morph_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, EROSION);
  }
}
// Template instantiations for specific types
template void erosion_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                     int, int, int, float, bool);
template void erosion_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, int*, int, int, int, float,
                                              bool);
template void erosion_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                       int*, int, int, int, float, bool);

/**
 * @brief Perform dilation operation on a grayscale image.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
template <typename dtype>
void dilation_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                        int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {

  if (gpu) {
    int ncopies = 2;
    int flag_chain = 0;
    chunkedExecutorKernel(morph_grayscale_on_device<dtype>, ncopies, gpuMemory, flag_chain,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, DILATION);
  }

  else {
    morph_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, DILATION);
  }
}
// Template instantiations for specific types
template void dilation_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                      int, int, int, float, bool);
template void dilation_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                               const int, const int, int*, int, int, int, float,
                                               bool);
template void dilation_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                        int*, int, int, int, float, bool);

/**
 * @brief Perform closing operation on a grayscale image.
 *
 * Closing is defined as a dilation followed by an erosion.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
template <typename dtype>
void closing_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {
  MorphChain closing = {DILATION, EROSION};

  if (gpu) {
    int ncopies = 3;
    int flag_chain = 1;
    chunkedExecutorKernel(morph_chain_grayscale_on_device<dtype>, ncopies, gpuMemory, flag_chain,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, closing);

  } else {
    morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                  kernel_ysize, kernel_zsize, closing);
  }
}
// Template instantiations for specific types
template void closing_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                     int, int, int, float, bool);
template void closing_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, int*, int, int, int, float,
                                              bool);
template void closing_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                       int*, int, int, int, float, bool);

/**
 * @brief Perform opening operation on a grayscale image.
 *
 * Opening is defined as an erosion followed by a dilation.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
template <typename dtype>
void opening_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {

  MorphChain opening = {EROSION, DILATION};

  if (gpu) {
    int ncopies = 3;
    int flag_chain = 1;
    chunkedExecutorKernel(morph_chain_grayscale_on_device<dtype>, ncopies, gpuMemory, flag_chain,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, opening);
  } else {
    morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                  kernel_ysize, kernel_zsize, opening);
  }
}
// Template instantiations for specific types
template void opening_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                     int, int, int, float, bool);
template void opening_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, int*, int, int, int, float,
                                              bool);
template void opening_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                       int*, int, int, int, float, bool);

/**
 * @brief Perform geodesic erosion operation on the entire image using the GPU. This function is
 * meant to be called from host and slide the morph_grayscale kerel function through all pixels.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 */
template <typename dtype>
void geodesic_erosion_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                const int xsize, const int ysize, const int zsize,
                                const int flag_verbose, float gpuMemory, bool gpu) {
  if (gpu) {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_grayscale_on_device<dtype>, ncopies, gpuMemory,
                            hostImage, hostMask, hostOutput, xsize, ysize, zsize, flag_verbose,
                            EROSION);

  } else {
    geodesic_morph_grayscale_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, EROSION);
  }
}
template void geodesic_erosion_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                              const int, float, bool);
template void geodesic_erosion_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                       const int, const int, const int, const int,
                                                       float, bool);
template void geodesic_erosion_grayscale<float>(float*, float*, float*, const int, const int,
                                                const int, const int, float, bool);

/**
 * @brief Perform geodesic dilation operation on the entire image using the GPU. This function is
 * meant to be called from host and slide the morph_grayscale kerel function through all pixels.
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 */
template <typename dtype>
void geodesic_dilation_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                 const int xsize, const int ysize, const int zsize,
                                 const int flag_verbose, float gpuMemory, bool gpu) {
  if (gpu) {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_grayscale_on_device<dtype>, ncopies, gpuMemory,
                            hostImage, hostMask, hostOutput, xsize, ysize, zsize, flag_verbose,
                            DILATION);
  } else {
    geodesic_morph_grayscale_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize,
                                     DILATION);
  }
}
template void geodesic_dilation_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                               const int, float, bool);
template void geodesic_dilation_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                        const int, const int, const int, const int,
                                                        float, bool);
template void geodesic_dilation_grayscale<float>(float*, float*, float*, const int, const int,
                                                 const int, const int, float, bool);

template <typename dtype>
void reconstruction_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              MorphOp operation, bool gpu) {

  if (gpu) {
    reconstruction_grayscale_on_device(hostImage, hostMask, hostOutput, xsize, ysize, zsize,
                                       operation, flag_verbose);
  } else {
    reconstruction_grayscale_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize,
                                     operation);
  }
}
template void reconstruction_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                            const int, MorphOp, bool);
template void reconstruction_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                     const int, const int, const int, const int,
                                                     MorphOp, bool);
template void reconstruction_grayscale<float>(float*, float*, float*, const int, const int,
                                              const int, const int, MorphOp, bool);

template <typename dtype>
void bottom_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {

  if (gpu) {
    int ncopies = 3;
    int flag_chain = 1;
    chunkedExecutorKernel(bottom_hat_on_device<dtype>, ncopies, gpuMemory, flag_chain, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  } else {
    // bottom hat operation
    bottom_hat_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                       kernel_ysize, kernel_zsize);
  }
}
template void bottom_hat<int>(int*, int*, const int, const int, const int, const int, int*, int,
                              int, int, float, bool);
template void bottom_hat<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                       const int, const int, int*, int, int, int, float, bool);
template void bottom_hat<float>(float*, float*, const int, const int, const int, const int, int*,
                                int, int, int, float, bool);

template <typename dtype>
void top_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize, const int zsize,
             const int flag_verbose, int* kernel, int kernel_xsize, int kernel_ysize,
             int kernel_zsize, float gpuMemory, bool gpu) {

  if (gpu) {
    int ncopies = 3;
    int flag_chain = 1;
    chunkedExecutorKernel(top_hat_on_device<dtype>, ncopies, gpuMemory, flag_chain, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  } else {
    // bottom hat operation
    top_hat_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                    kernel_zsize);
  }
}
template void top_hat<int>(int*, int*, const int, const int, const int, const int, int*, int, int,
                           int, float, bool);
template void top_hat<unsigned int>(unsigned int*, unsigned int*, const int, const int, const int,
                                    const int, int*, int, int, int, float, bool);
template void top_hat<float>(float*, float*, const int, const int, const int, const int, int*, int,
                             int, int, float, bool);

template <typename dtype>
void top_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                            int kernel_ysize, int kernel_zsize, bool gpu) {

  if (gpu) {
    top_hat_reconstruction_on_device(hostImage, hostOutput, xsize, ysize, zsize, flag_verbose,
                                     kernel, kernel_xsize, kernel_ysize, kernel_zsize);
  } else {
    top_hat_reconstruction_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                   kernel_ysize, kernel_zsize);
  }
}
template void top_hat_reconstruction<int>(int*, int*, const int, const int, const int, const int,
                                          int*, int, int, int, bool);
template void top_hat_reconstruction<unsigned int>(unsigned int*, unsigned int*, const int,
                                                   const int, const int, const int, int*, int, int,
                                                   int, bool);
template void top_hat_reconstruction<float>(float*, float*, const int, const int, const int,
                                            const int, int*, int, int, int, bool);