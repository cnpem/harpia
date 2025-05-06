#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/fill_holes.h"
#include "../../../include/tests/morphology/test_fill_holes.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_util.h"

void test_fill_holes_on_device(const std::string& filename, const int xsize, const int ysize,
                               const int zsize, int padding, float memoryOccupancy, int ngpus, const int flag_check,
                               const int flag_verbose) {

  printf("\nTest fill holes on device\n");

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)calloc(size, sizeof(int));

  read_input(host_A, filename, size, flag_verbose);

  int ncopies = 3;
  chunkedExecutorFillHoles(fill_holes_on_device<int>, ncopies, memoryOccupancy, ngpus, host_A, device_ref,
                           padding, xsize, ysize, zsize, flag_verbose);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)calloc(size, sizeof(int));
    // Perform binary morphology on host for comparison
    // fill_holes_on_host(host_A, host_ref, xsize, ysize, zsize);
    fill_holes_on_device(host_A, host_ref, xsize, ysize, zsize, flag_verbose);

    check_result(device_ref, host_ref, xsize, ysize, zsize);
    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}

/**
 * @brief Tests the custom binary erosion operation and compares its result with OpenCV's
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
void test_fill_holes_on_host(const std::string& filename, const int xsize, const int ysize,
                             const int zsize, const int flag_verbose) {

  printf("\nTest fill holes on host\n");

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);

  if (flag_verbose) {
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);
  }

  int *host_A, *host_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  host_ref = (int*)calloc(size, sizeof(int));

  // Set input data
  read_input(host_A, filename, size, flag_verbose);

  // Perform binary morphology on host
  fill_holes_on_host(host_A, host_ref, xsize, ysize, zsize);

  show_image_3D(host_A, xsize, ysize, zsize, "Input Image");
  show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");
  cv::waitKey(0);  // Needed for the show_image_3D() calls

  // Free host memory
  free(host_A);
  free(host_ref);
}
