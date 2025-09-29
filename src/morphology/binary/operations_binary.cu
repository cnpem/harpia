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
                    int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {
  if (ngpus == 0) {
    morph_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
                         kernel_ysize, kernel_zsize, EROSION);
  } else {
    int ncopies = 2;
    int operations = 1;
    chunkedExecutorKernel(morph_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, EROSION);                          
  }
}
template void erosion_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                  int, int, float, int);
template void erosion_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                           const int, const int, int*, int, int, int, float, int);
template void erosion_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, int);
template void erosion_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, int);

template <typename dtype>
void dilation_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                     const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                     int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {
  if (ngpus == 0) {
    morph_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, DILATION);
  } else {
    int ncopies = 2;
    int operations = 1;
    chunkedExecutorKernel(morph_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize, DILATION);
  }
}
template void dilation_binary<int>(int*, int*, const int, const int, const int, const int, int*,
                                   int, int, int, float, int);
template void dilation_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                            const int, const int, int*, int, int, int, float, int);
template void dilation_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, int);
template void dilation_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                        const int, int*, int, int, int, float, int);

template <typename dtype>
void closing_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {

  MorphChain closing = {DILATION, EROSION};
  if (ngpus == 0) {
    morph_chain_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, closing);
  } else {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(morph_chain_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, closing);
  }
}
template void closing_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                  int, int, float, int);
template void closing_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                           const int, const int, int*, int, int, int, float, int);
template void closing_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, int);
template void closing_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, int);

template <typename dtype>
void opening_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {
  MorphChain opening = {EROSION, DILATION};

  if (ngpus == 0) {
    morph_chain_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, opening);
  } else {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(morph_chain_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, opening);
  }
}
template void opening_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                  int, int, float, int);
template void opening_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                           const int, const int, int*, int, int, int, float, int);
template void opening_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, int);
template void opening_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                       const int, int*, int, int, int, float, int);

template <typename dtype>
void smooth_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                   const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                   int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {

  if (ngpus == 0) {
    smooth_binary_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize);
  } else {
    int ncopies = 2;
    int operations = 4;
    chunkedExecutorKernel(smooth_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  }
}
template void smooth_binary<int>(int*, int*, const int, const int, const int, const int, int*, int,
                                 int, int, float, int);
template void smooth_binary<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                          const int, const int, int*, int, int, int, float, int);
template void smooth_binary<int16_t>(int16_t*, int16_t*, const int, const int, const int, const int,
                                     int*, int, int, int, float, int);
template void smooth_binary<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int,
                                      const int, int*, int, int, int, float, int);

template <typename dtype>
void geodesic_erosion_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                             const int ysize, const int zsize, const int flag_verbose,
                             float gpuMemory, int ngpus) {
  if (ngpus == 0) {
    geodesic_morph_binary_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, EROSION);    
  } else {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, hostImage,
                            hostMask, hostOutput, xsize, ysize, zsize, flag_verbose, EROSION);
  }
}
template void geodesic_erosion_binary<int>(int*, int*, int*, const int, const int, const int,
                                           const int, float, int);
template void geodesic_erosion_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                    const int, const int, const int, const int,
                                                    float, int);
template void geodesic_erosion_binary<int16_t>(int16_t*, int16_t*, int16_t*, const int, const int,
                                               const int, const int, float, int);
template void geodesic_erosion_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int,
                                                const int, const int, const int, float, int);

template <typename dtype>
void geodesic_dilation_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              float gpuMemory, int ngpus) {

  if (ngpus == 0) {
    geodesic_morph_binary_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, DILATION);
  }
  else {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_binary_on_device<dtype>, ncopies, gpuMemory, ngpus, hostImage,
                            hostMask, hostOutput, xsize, ysize, zsize, flag_verbose, DILATION);
  }
}
template void geodesic_dilation_binary<int>(int*, int*, int*, const int, const int, const int,
                                            const int, float, int);
template void geodesic_dilation_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                     const int, const int, const int, const int,
                                                     float, int);
template void geodesic_dilation_binary<int16_t>(int16_t*, int16_t*, int16_t*, const int, const int,
                                                const int, const int, float, int);
template void geodesic_dilation_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int,
                                                 const int, const int, const int, float, int);

template <typename dtype>
void reconstruction_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                           const int ysize, const int zsize, const int flag_verbose,
                           MorphOp operation, int ngpus) {

 // TODO: add a check if there is sufficiet memory on the gpu to perform the operation. Since it is 
 // a convergence operation it cannot be broken in chunks.

 if (ngpus == 0) {
    reconstruction_binary_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, operation);
  } else {
    reconstruction_binary_on_device(hostImage, hostMask, hostOutput, xsize, ysize, zsize, operation,
      flag_verbose);
  }
}
template void reconstruction_binary<int>(int*, int*, int*, const int, const int, const int,
                                         const int, MorphOp, int);
template void reconstruction_binary<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                  const int, const int, const int, const int,
                                                  MorphOp, int);
template void reconstruction_binary<int16_t>(int16_t*, int16_t*, int16_t*, const int, const int,
                                             const int, const int, MorphOp, int);
template void reconstruction_binary<uint16_t>(uint16_t*, uint16_t*, uint16_t*, const int, const int,
                                              const int, const int, MorphOp, int);

template <typename dtype>
void fill_holes(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, int padding, const int flag_verbose, float gpuMemory, int ngpus) {

 // TODO: add a check if there is sufficiet memory on the gpu to perform the operation. Since it is 
 // a convergence operation it cannot be broken in chunks.

  if (ngpus == 0) {
    fill_holes_on_host(hostImage, hostOutput, xsize, ysize, zsize);
  } else {
    int ncopies = 3;
    
    chunkedExecutorFillHoles(fill_holes_on_device<dtype>, ncopies, gpuMemory, ngpus, hostImage, hostOutput,
                             padding, xsize, ysize, zsize, flag_verbose);
  }
}
template void fill_holes<int>(int*, int*, const int, const int, const int, int, const int, float, 
                              int);
template void fill_holes<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                       const int, int, const int, float, int);
template void fill_holes<int16_t>(int16_t*, int16_t*, const int, const int, const int, int,
                                  const int, float, int);
template void fill_holes<uint16_t>(uint16_t*, uint16_t*, const int, const int, const int, int, 
                                   const int, float, int);
