#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/fill_holes.h"
#include "../../../include/morphology/geodesic_morph_binary.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/operations_binary.h"
#include "../../../include/morphology/reconstruction_binary.h"
#include "../../../include/morphology/smooth_binary.h"

/**
 * @brief Performs binary erosion on the input image.
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
void erosion_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {
  if (gpu) {
    int ncopies = 2;
    int operations = 1;
    chunkedExecutorKernel(morph_binary_on_device<dtype>, ncopies, gpuMemory, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, EROSION);
  } else {
    morph_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                         kernel_ysize, kernel_zsize, EROSION);
  }
}
template void erosion_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                  int, int, float, bool);
template void erosion_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                           const int, const int, int*, int, int, int, float, bool);
template void erosion_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);
template void erosion_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, bool);
template void erosion_binary<int8_t>(int8_t*, int8_t*, const int, const int, const int, const int,
                                     int*, int, int, int, float, bool);
template void erosion_binary<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);

/**
 * @brief Performs binary dilation on the input image.
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
void dilation_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                     const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                     int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {
  if (gpu) {
    int ncopies = 2;
    int operations = 1;
    chunkedExecutorKernel(morph_binary_on_device<dtype>, ncopies, gpuMemory, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, DILATION);
  } else {
    morph_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                         kernel_ysize, kernel_zsize, DILATION);
  }
}
template void dilation_binary<int>(int*, int*, const int, const int, const int, const int, int*,
                                   int, int, int, float, bool);
template void dilation_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                            const int, const int, int*, int, int, int, float, bool);
template void dilation_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, bool);
template void dilation_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                        const int, int*, int, int, int, float, bool);
template void dilation_binary<int8_t>(int8_t*, int8_t*, const int, const int, const int, const int,
                                      int*, int, int, int, float, bool);
template void dilation_binary<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, bool);

/**
 * @brief Performs binary closing on the input image.
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
void closing_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {

  MorphChain closing = {DILATION, EROSION};
  if (gpu) {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(morph_chain_binary_on_device<dtype>, ncopies, gpuMemory, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, closing);
  } else {
    morph_chain_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, closing);
  }
}
template void closing_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                  int, int, float, bool);
template void closing_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                           const int, const int, int*, int, int, int, float, bool);
template void closing_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);
template void closing_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, bool);
template void closing_binary<int8_t>(int8_t*, int8_t*, const int, const int, const int, const int,
                                     int*, int, int, int, float, bool);
template void closing_binary<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);

/**
 * @brief Performs binary openig on the input image.
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
void opening_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {
  MorphChain opening = {EROSION, DILATION};

  if (gpu) {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(morph_chain_binary_on_device<dtype>, ncopies, gpuMemory, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, opening);
  } else {
    morph_chain_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, opening);
  }
}
template void opening_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                  int, int, float, bool);
template void opening_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                           const int, const int, int*, int, int, int, float, bool);
template void opening_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);
template void opening_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, bool);
template void opening_binary<int8_t>(int8_t*, int8_t*, const int, const int, const int, const int,
                                     int*, int, int, int, float, bool);
template void opening_binary<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);

template <typename dtype>
void smooth_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                   const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                   int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu) {

  if (gpu) {
    int ncopies = 2;
    int operations = 4;
    chunkedExecutorKernel(smooth_binary_on_device<dtype>, ncopies, gpuMemory, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  } else {
    smooth_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  }
}
template void smooth_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                 int, int, float, bool);
template void smooth_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                          const int, const int, int*, int, int, int, float, bool);
template void smooth_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int, const int,
                                     int*, int, int, int, float, bool);
template void smooth_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, bool);
template void smooth_binary<int8_t>(int8_t*, int8_t*, const int, const int, const int, const int,
                                    int*, int, int, int, float, bool);
template void smooth_binary<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int, const int,
                                     int*, int, int, int, float, bool);

/**
 * @brief Perform geodesic erosion operation on the entire image using the GPU. This function is
 * meant to be called from host and slide the morph_binary kerel function through all pixels.
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
void geodesic_erosion_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                             const int ysize, const int zsize, const int flag_verbose,
                             float gpuMemory, bool gpu) {
  if (gpu) {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_binary_on_device<dtype>, ncopies, gpuMemory, hostImage,
                            hostMask, hostOutput, xsize, ysize, zsize, flag_verbose, EROSION);

  } else {
    geodesic_morph_binary_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, EROSION);
  }
}
template void geodesic_erosion_binary<int>(int*, int*, int*, const int, const int, const int,
                                           const int, float, bool);
template void geodesic_erosion_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                    const int, const int, const int, const int,
                                                    float, bool);
template void geodesic_erosion_binary<int16_t>(int16_t*, int16_t*, int16_t*, const int, const int,
                                               const int, const int, float, bool);
template void geodesic_erosion_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int,
                                                const int, const int, const int, float, bool);
template void geodesic_erosion_binary<int8_t>(int8_t*, int8_t*, int8_t*, const int, const int,
                                              const int, const int, float, bool);
template void geodesic_erosion_binary<uint8_t>(uint8_t*, uint8_t*, uint8_t*, const int, const int,
                                               const int, const int, float, bool);

/**
 * @brief Perform geodesic dilation operation on the entire image using the GPU. This function is
 * meant to be called from host and slide the morph_binary kerel function through all pixels.
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
void geodesic_dilation_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              float gpuMemory, bool gpu) {

  if (gpu) {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_binary_on_device<dtype>, ncopies, gpuMemory, hostImage,
                            hostMask, hostOutput, xsize, ysize, zsize, flag_verbose, DILATION);
  } else {
    geodesic_morph_binary_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, DILATION);
  }
}
template void geodesic_dilation_binary<int>(int*, int*, int*, const int, const int, const int,
                                            const int, float, bool);
template void geodesic_dilation_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                     const int, const int, const int, const int,
                                                     float, bool);
template void geodesic_dilation_binary<int16_t>(int16_t*, int16_t*, int16_t*, const int, const int,
                                                const int, const int, float, bool);
template void geodesic_dilation_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int,
                                                 const int, const int, const int, float, bool);
template void geodesic_dilation_binary<int8_t>(int8_t*, int8_t*, int8_t*, const int, const int,
                                               const int, const int, float, bool);
template void geodesic_dilation_binary<uint8_t>(uint8_t*, uint8_t*, uint8_t*, const int, const int,
                                                const int, const int, float, bool);

template <typename dtype>
void reconstruction_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                           const int ysize, const int zsize, const int flag_verbose,
                           MorphOp operation, bool gpu) {

  if (gpu) {
    reconstruction_binary_on_device(hostImage, hostMask, hostOutput, xsize, ysize, zsize, operation,
                                    flag_verbose);
  } else {
    reconstruction_binary_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, operation);
  }
}
template void reconstruction_binary<int>(int*, int*, int*, const int, const int, const int,
                                         const int, MorphOp, bool);
template void reconstruction_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                  const int, const int, const int, const int,
                                                  MorphOp, bool);
template void reconstruction_binary<int16_t>(int16_t*, int16_t*, int16_t*, const int, const int,
                                             const int, const int, MorphOp, bool);
template void reconstruction_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int, const int,
                                              const int, const int, MorphOp, bool);
template void reconstruction_binary<int8_t>(int8_t*, int8_t*, int8_t*, const int, const int,
                                            const int, const int, MorphOp, bool);
template void reconstruction_binary<uint8_t>(uint8_t*, uint8_t*, uint8_t*, const int, const int,
                                             const int, const int, MorphOp, bool);

template <typename dtype>
void fill_holes(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, const int flag_verbose, bool gpu) {

  if (gpu) {
    fill_holes_on_device(hostImage, hostOutput, xsize, ysize, zsize, flag_verbose);
  } else {
    fill_holes_on_host(hostImage, hostOutput, xsize, ysize, zsize);
  }
}
// Template instantiations for specific types
template void fill_holes<int>(int*, int*, const int, const int, const int, const int, bool);
template void fill_holes<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                       const int, const int, bool);
template void fill_holes<int16_t>(int16_t*, int16_t*, const int, const int, const int, const int,
                                  bool);
template void fill_holes<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int, const int,
                                   bool);
template void fill_holes<int8_t>(int8_t*, int8_t*, const int, const int, const int, const int,
                                 bool);
template void fill_holes<uint8_t>(uint8_t*, uint8_t*, const int, const int, const int, const int,
                                  bool);
