#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/operations_grayscale.h"

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
void erosion_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                 int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  morph_grayscale_on_device(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, EROSION, flag_verbose);
}
// Template instantiations for specific types
template void erosion_grayscale_on_device<int>(int*, int*, const int, const int, const int, int*,
                                               int, int, int, const int);
template void erosion_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                        const int, const int, int*, int, int, int,
                                                        const int);
template void erosion_grayscale_on_device<float>(float*, float*, const int, const int, const int,
                                                 int*, int, int, int, const int);

template <typename dtype>
void erosion_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, int* kernel, int kernel_xsize,
                               int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  morph_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, EROSION);
}
// Template instantiations for specific types
template void erosion_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*, int,
                                             int, int, const int);
template void erosion_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                      const int, const int, int*, int, int, int,
                                                      const int);
template void erosion_grayscale_on_host<float>(float*, float*, const int, const int, const int,
                                               int*, int, int, int, const int);

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
void dilation_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                  const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                  int kernel_ysize, int kernel_zsize,

                                  const int flag_verbose) {
  morph_grayscale_on_device(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, DILATION, flag_verbose);
}
// Template instantiations for specific types
template void dilation_grayscale_on_device<int>(int*, int*, const int, const int, const int, int*,
                                                int, int, int, const int);
template void dilation_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                         const int, const int, int*, int, int, int,
                                                         const int);
template void dilation_grayscale_on_device<float>(float*, float*, const int, const int, const int,
                                                  int*, int, int, int, const int);

template <typename dtype>
void dilation_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                                const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  morph_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, DILATION);
}
// Template instantiations for specific types
template void dilation_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*,
                                              int, int, int, const int);
template void dilation_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                       const int, const int, int*, int, int, int,
                                                       const int);
template void dilation_grayscale_on_host<float>(float*, float*, const int, const int, const int,
                                                int*, int, int, int, const int);

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
void closing_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                 int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  MorphChain closing = {DILATION, EROSION};

  morph_chain_grayscale_on_device(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                  kernel_ysize, kernel_zsize, closing, flag_verbose);
}
// Template instantiations for specific types
template void closing_grayscale_on_device<int>(int*, int*, const int, const int, const int, int*,
                                               int, int, int, const int);
template void closing_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                        const int, const int, int*, int, int, int,
                                                        const int);
template void closing_grayscale_on_device<float>(float*, float*, const int, const int, const int,
                                                 int*, int, int, int, const int);

template <typename dtype>
void closing_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, int* kernel, int kernel_xsize,
                               int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  MorphChain closing = {DILATION, EROSION};

  morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                kernel_ysize, kernel_zsize, closing);
}
// Template instantiations for specific types
template void closing_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*, int,
                                             int, int, const int);
template void closing_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                      const int, const int, int*, int, int, int,
                                                      const int);
template void closing_grayscale_on_host<float>(float*, float*, const int, const int, const int,
                                               int*, int, int, int, const int);

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
void opening_grayscale_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, int* kernel, int kernel_xsize,
                                 int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  MorphChain opening = {EROSION, DILATION};

  morph_chain_grayscale_on_device(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                  kernel_ysize, kernel_zsize, opening, flag_verbose);
}
// Template instantiations for specific types
template void opening_grayscale_on_device<int>(int*, int*, const int, const int, const int, int*,
                                               int, int, int, const int);
template void opening_grayscale_on_device<unsigned int>(unsigned int*, unsigned int*, const int,
                                                        const int, const int, int*, int, int, int,
                                                        const int);
template void opening_grayscale_on_device<float>(float*, float*, const int, const int, const int,
                                                 int*, int, int, int, const int);

template <typename dtype>
void opening_grayscale_on_host(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, int* kernel, int kernel_xsize,
                               int kernel_ysize, int kernel_zsize, const int flag_verbose) {
  MorphChain opening = {EROSION, DILATION};

  morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                                kernel_ysize, kernel_zsize, opening);
}
// Template instantiations for specific types
template void opening_grayscale_on_host<int>(int*, int*, const int, const int, const int, int*, int,
                                             int, int, const int);
template void opening_grayscale_on_host<unsigned int>(unsigned int*, unsigned int*, const int,
                                                      const int, const int, int*, int, int, int,
                                                      const int);
template void opening_grayscale_on_host<float>(float*, float*, const int, const int, const int,
                                               int*, int, int, int, const int);