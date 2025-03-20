#include <cstdint>  // For uint16_t, unsigned int
#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/fill_holes.h"
#include "../../../include/morphology/geodesic_morph_binary.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/operations_binary.h"
#include "../../../include/morphology/reconstruction_binary.h"
#include "../../../include/morphology/smooth_binary.h"

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

 // TODO: add a check if there is sufficiet memory on the gpu to perform the operation. Since it is 
 // a convergence operation it cannot be broken in chunks.
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

template <typename dtype>
void fill_holes(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, int padding, const int flag_verbose, float gpuMemory, bool gpu) {

  if (gpu) {
    int ncopies = 3;
    chunkedExecutorFillHoles(fill_holes_on_device<dtype>, ncopies, gpuMemory, hostImage, hostOutput,
                             padding, xsize, ysize, zsize, flag_verbose);
  } else {
    fill_holes_on_host(hostImage, hostOutput, xsize, ysize, zsize);
  }
}
template void fill_holes<int>(int*, int*, const int, const int, const int, int, const int, float, 
                              bool);
template void fill_holes<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                       const int, int, const int, float, bool);
template void fill_holes<int16_t>(int16_t*, int16_t*, const int, const int, const int, int,
                                  const int, float, bool);
template void fill_holes<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int, int, 
                                   const int, float, bool);
