#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/geodesic_morph_grayscale.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/tests/morphology/test_geodesic_morph_grayscale.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_util.h"

/**
 * @brief Tests the grayscale morphological operations on the GPU.
 * 
 * @param filename The name of the input file.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the structuring element kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param operation The morphological operation to perform (e.g., erosion, dilation).
 * @param flag_check If set, the results will be checked for correctness.
 * @param flag_verbose If set, additional information will be printed.
 */
void test_geodesic_morph_grayscale_on_device(const std::string& filename, const int xsize,
                                             const int ysize, const int zsize, MorphOp operation,
                                             float memoryOccupancy, int ngpus, const int flag_check,
                                             const int flag_verbose, const int flag_float) {

  printf("\nTest grayscale geodesic %s on device\n", (operation ? "dilation" : "erosion"));

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(float);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *hostMarker, *device_ref, *hostMask;  // Pointers for host memory
  hostMarker = (float*)malloc(nBytes);
  hostMask = (float*)malloc(nBytes);
  device_ref = (float*)calloc(size, sizeof(float));

  // Set input data
  read_input(hostMask, filename, size, flag_verbose, flag_float);

  //create marker image
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * 27);

  int ncopies = 2;
  int operations = 1;
  if (operation == EROSION) {
    get_structuring_element_3D(kernel, 3, 3, 3);
    chunkedExecutorKernel(morph_grayscale_on_device<float>, ncopies, memoryOccupancy, ngpus, operations,
                          hostMask, hostMarker, xsize, ysize, zsize, flag_verbose, kernel, 3, 3, 3,
                          DILATION);
  } else {
    horizontal_line_kernel(kernel);
    chunkedExecutorKernel(morph_grayscale_on_device<float>, ncopies, memoryOccupancy, ngpus, operations,
                          hostMask, hostMarker, xsize, ysize, zsize, flag_verbose, kernel, 3, 3, 3,
                          EROSION);
  }
  ncopies = 3;
  chunkedExecutorGeodesic(geodesic_morph_grayscale_on_device<float>, ncopies, memoryOccupancy, ngpus,
                          hostMarker, hostMask, device_ref, xsize, ysize, zsize, flag_verbose,
                          operation);
  if (flag_check) {
    float* host_ref;
    host_ref = (float*)calloc(size, sizeof(float));
    geodesic_morph_grayscale_on_host(hostMarker, hostMask, host_ref, xsize, ysize, zsize,
                                     operation);
    check_result(host_ref, device_ref, xsize, ysize, zsize);
    free(host_ref);
  }

  free(hostMarker);
  free(hostMask);
  free(device_ref);
  free(kernel);
}

/**
 * @brief Tests the custom grayscale erosion operation and compares its result with OpenCV's 
 * implementation.
 * 
 * @param filename The name of the input file.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the structuring element kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param operation The morphological operation to perform (e.g., erosion, dilation).
 * @param flag_show If set, the input and result images will be displayed.
 * @param flag_check If set, the results will be checked for correctness.
 * @param flag_verbose If set, additional information will be printed.
 */
void test_geodesic_morph_grayscale_on_host(const std::string& filename, const int xsize,
                                           const int ysize, const int zsize, MorphOp operation,
                                           const int flag_verbose, const int flag_float) {

  printf("\nTest grayscale geodesic %s on host\n", (operation ? "dilation" : "erosion"));

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  size_t nBytes = size * sizeof(float);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *hostMarker, *host_ref, *hostMask;  // Pointers for host memory
  hostMarker = (float*)malloc(nBytes);
  hostMask = (float*)malloc(nBytes);
  host_ref = (float*)calloc(size, sizeof(float));

  // Set input data
  read_input(hostMask, filename, size, flag_verbose, flag_float);

  //create marker image
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * 27);

  if (operation == EROSION) {
    get_structuring_element_3D(kernel, 3, 3, 3);
    morph_grayscale_on_host(hostMask, hostMarker, xsize, ysize, zsize, kernel, 3, 3, 3, DILATION);
  } else {
    horizontal_line_kernel(kernel);
    morph_grayscale_on_host(hostMask, hostMarker, xsize, ysize, zsize, kernel, 3, 3, 3, EROSION);
  }

  geodesic_morph_grayscale_on_host(hostMarker, hostMask, host_ref, xsize, ysize, zsize, operation);
  show_image_3D(hostMarker, xsize, ysize, 1, "Marker");
  show_image_3D(hostMask, xsize, ysize, 1, "Mask");
  show_image_3D(host_ref, xsize, ysize, 1, "Host");
  show_image_3D(host_ref, xsize, ysize, 1, "Device");

  cv::waitKey(0);

  free(hostMarker);
  free(hostMask);
  free(host_ref);
  free(kernel);
}