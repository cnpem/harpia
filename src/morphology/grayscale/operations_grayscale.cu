#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/bottom_hat.h"
#include "../../../include/morphology/bottom_hat_reconstruction.h"
#include "../../../include/morphology/geodesic_morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/operations_grayscale.h"
#include "../../../include/morphology/reconstruction_grayscale.h"
#include "../../../include/morphology/top_hat.h"
#include "../../../include/morphology/top_hat_reconstruction.h"

template <typename dtype>
void erosion_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {
  if (ngpus == 0) {
    morph_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, EROSION);
  }
  else {
    int ncopies = 2;
    int operations = 1;
    chunkedExecutorKernel(morph_grayscale_on_device<dtype>, ncopies, gpuMemory, ngpus, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, EROSION);
  }
}
template void erosion_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                     int, int, int, float, int);
template void erosion_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, int*, int, int, int, float,
                                              int);
template void erosion_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                       int*, int, int, int, float, int);

template <typename dtype>
void dilation_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                        int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {

  if (ngpus == 0) {
    morph_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, DILATION);
  }
  else {
    int ncopies = 2;
    int operations = 1;
    chunkedExecutorKernel(morph_grayscale_on_device<dtype>, ncopies, gpuMemory, ngpus, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, DILATION);
  }
}
template void dilation_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                      int, int, int, float, int);
template void dilation_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                               const int, const int, int*, int, int, int, float,
                                               int);
template void dilation_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                        int*, int, int, int, float, int);

template <typename dtype>
void closing_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {
  MorphChain closing = {DILATION, EROSION};

  if (ngpus == 0) {
    morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, closing);
  } else {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(morph_chain_grayscale_on_device<dtype>, ncopies, gpuMemory, ngpus, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, closing);
  }
}
template void closing_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                     int, int, int, float, int);
template void closing_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, int*, int, int, int, float,
                                              int);
template void closing_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                       int*, int, int, int, float, int);

template <typename dtype>
void opening_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {

  MorphChain opening = {EROSION, DILATION};

  if (ngpus == 0) {
    morph_chain_grayscale_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize, opening);
  } else {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(morph_chain_grayscale_on_device<dtype>, ncopies, gpuMemory, ngpus, operations,
                          hostImage, hostOutput, xsize, ysize, zsize, flag_verbose, kernel,
                          kernel_xsize, kernel_ysize, kernel_zsize, opening);
  }
}
template void opening_grayscale<int>(int*, int*, const int, const int, const int, const int, int*,
                                     int, int, int, float, int);
template void opening_grayscale<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                              const int, const int, int*, int, int, int, float,
                                              int);
template void opening_grayscale<float>(float*, float*, const int, const int, const int, const int,
                                       int*, int, int, int, float, int);

template <typename dtype>
void geodesic_erosion_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                const int xsize, const int ysize, const int zsize,
                                const int flag_verbose, float gpuMemory, int ngpus) {
  if (ngpus == 0) {
    geodesic_morph_grayscale_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize, EROSION);
  } else {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_grayscale_on_device<dtype>, ncopies, gpuMemory, ngpus,
                            hostImage, hostMask, hostOutput, xsize, ysize, zsize, flag_verbose,
                            EROSION);
  }
}
template void geodesic_erosion_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                              const int, float, int);
template void geodesic_erosion_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                       const int, const int, const int, const int,
                                                       float, int);
template void geodesic_erosion_grayscale<float>(float*, float*, float*, const int, const int,
                                                const int, const int, float, int);

template <typename dtype>
void geodesic_dilation_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                 const int xsize, const int ysize, const int zsize,
                                 const int flag_verbose, float gpuMemory, int ngpus) {
  if (ngpus == 0) {
    geodesic_morph_grayscale_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize,
      DILATION);
  } else {
    int ncopies = 3;
    chunkedExecutorGeodesic(geodesic_morph_grayscale_on_device<dtype>, ncopies, gpuMemory, ngpus,
                            hostImage, hostMask, hostOutput, xsize, ysize, zsize, flag_verbose,
                            DILATION);
  }
}
template void geodesic_dilation_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                               const int, float, int);
template void geodesic_dilation_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                        const int, const int, const int, const int,
                                                        float, int);
template void geodesic_dilation_grayscale<float>(float*, float*, float*, const int, const int,
                                                 const int, const int, float, int);

template <typename dtype>
void reconstruction_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              MorphOp operation, int ngpus) {

  if (ngpus == 0) {
    reconstruction_grayscale_on_host(hostImage, hostMask, hostOutput, xsize, ysize, zsize,
      operation);
  } else {
    reconstruction_grayscale_on_device(hostImage, hostMask, hostOutput, xsize, ysize, zsize,
      operation, flag_verbose);
  }
}
template void reconstruction_grayscale<int>(int*, int*, int*, const int, const int, const int,
                                            const int, MorphOp, int);
template void reconstruction_grayscale<unsigned int>(unsigned int*, unsigned int*, unsigned int*,
                                                     const int, const int, const int, const int,
                                                     MorphOp, int);
template void reconstruction_grayscale<float>(float*, float*, float*, const int, const int,
                                              const int, const int, MorphOp, int);

template <typename dtype>
void bottom_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus) {

  if (ngpus == 0) {
    // bottom hat operation
    bottom_hat_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize);
  } else {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(bottom_hat_on_device<dtype>, ncopies, gpuMemory, ngpus, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  }
}
template void bottom_hat<int>(int*, int*, const int, const int, const int, const int, int*, int,
                              int, int, float, int);
template void bottom_hat<unsigned int>(unsigned int*, unsigned int*, const int, const int,
                                       const int, const int, int*, int, int, int, float, int);
template void bottom_hat<float>(float*, float*, const int, const int, const int, const int, int*,
                                int, int, int, float, int);

template <typename dtype>
void top_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize, const int zsize,
             const int flag_verbose, int* kernel, int kernel_xsize, int kernel_ysize,
             int kernel_zsize, float gpuMemory, int ngpus) {

  if (ngpus == 0) {
    // bottom hat operation
    top_hat_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
      kernel_zsize);
  } else {
    int ncopies = 3;
    int operations = 2;
    chunkedExecutorKernel(top_hat_on_device<dtype>, ncopies, gpuMemory, ngpus, operations, hostImage,
                          hostOutput, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                          kernel_ysize, kernel_zsize);
  }
}
template void top_hat<int>(int*, int*, const int, const int, const int, const int, int*, int, int,
                           int, float, int);
template void top_hat<unsigned int>(unsigned int*, unsigned int*, const int, const int, const int,
                                    const int, int*, int, int, int, float, int);
template void top_hat<float>(float*, float*, const int, const int, const int, const int, int*, int,
                             int, int, float, int);

template <typename dtype>
void top_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                            int kernel_ysize, int kernel_zsize, int ngpus) {

  if (ngpus == 0) {
    top_hat_reconstruction_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel, kernel_xsize,
      kernel_ysize, kernel_zsize);
  } else {
    top_hat_reconstruction_on_device(hostImage, hostOutput, xsize, ysize, zsize, flag_verbose,
      kernel, kernel_xsize, kernel_ysize, kernel_zsize);
  }
}
template void top_hat_reconstruction<int>(int*, int*, const int, const int, const int, const int,
                                          int*, int, int, int, int);
template void top_hat_reconstruction<unsigned int>(unsigned int*, unsigned int*, const int,
                                                   const int, const int, const int, int*, int, int,
                                                   int, int);
template void top_hat_reconstruction<float>(float*, float*, const int, const int, const int,
                                            const int, int*, int, int, int, int);

template <typename dtype>
void bottom_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, const int flag_verbose,
                               int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize,
                               int ngpus) {

  if (ngpus == 0) {
    bottom_hat_reconstruction_on_host(hostImage, hostOutput, xsize, ysize, zsize, kernel,
      kernel_xsize, kernel_ysize, kernel_zsize);
  } else {
    bottom_hat_reconstruction_on_device(hostImage, hostOutput, xsize, ysize, zsize, flag_verbose,
      kernel, kernel_xsize, kernel_ysize, kernel_zsize);
  }
}
template void bottom_hat_reconstruction<int>(int*, int*, const int, const int, const int, const int,
                                             int*, int, int, int, int);
template void bottom_hat_reconstruction<unsigned int>(unsigned int*, unsigned int*, const int,
                                                      const int, const int, const int, int*, int,
                                                      int, int, int);
template void bottom_hat_reconstruction<float>(float*, float*, const int, const int, const int,
                                               const int, int*, int, int, int, int);