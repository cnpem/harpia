#include <stdio.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/subtraction.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_subtraction.h"
#include "../../../include/tests/morphology/test_util.h"

void test_subtraction_on_device(const std::string& filename, const std::string& filename2,
                                const int xsize, const int ysize, const int zsize,
                                float memoryOccupancy, const int flag_check,
                                const int flag_verbose) {
  int size = xsize * ysize * zsize;
  // set input dimension
  size_t nBytes = size * sizeof(float);

  if (flag_verbose)
    printf("Matrix size:   %d (%d.%d.%d) \n", size, xsize, ysize, zsize);

  float *host_A, *device_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  device_ref = (float*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(device_ref, 0, nBytes);

  read_input(host_A, filename, size, flag_verbose);
  read_input(device_ref, filename2, size, flag_verbose);

  // device erosion
  int ncopies = 3;
  chunkedExecutor(subtraction_on_device<float>, ncopies, memoryOccupancy, host_A, device_ref, xsize,
                  ysize, zsize, flag_verbose, flag_verbose);
  //show_image_3D(device_ref + 54 * xsize * ysize, xsize, ysize, 2, "device_ref");

  if (flag_check) {
    float* host_ref;
    host_ref = (float*)malloc(nBytes);
    memset(host_ref, 0, nBytes);
    read_input(host_ref, filename2, size, flag_verbose);

    // erosion
    subtraction_on_host(host_A, host_ref, size);
    //show_image_3D(host_ref + 54 * xsize * ysize, xsize, ysize, 5, "host_ref");
    //cv::waitKey(0);

    check_result(host_ref, device_ref, xsize, ysize, zsize);
    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}
void test_subtraction_on_host(const std::string& filename, const std::string& filename2,
                              const int xsize, const int ysize, const int zsize,
                              const int flag_show, const int flag_check, const int flag_verbose) {

  printf("\nTest subtraction on host\n");

  // set input dimension
  int size = xsize * ysize * zsize;

  size_t nBytes = size * sizeof(float);
  if (flag_verbose) {
    printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);
  }

  float *host_A, *host_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  host_ref = (float*)malloc(nBytes);

  // set input data
  memset(host_A, 1, nBytes);
  memset(host_ref, 1, nBytes);
  read_input(host_A, filename, size, flag_verbose);
  read_input(host_ref, filename, size, flag_verbose);
  if (flag_show) {
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image A");
    show_image_3D(host_ref, xsize, ysize, zsize, "Input Image B");
  }

  subtraction_on_host(host_A, host_ref, size);

  if (flag_show) {
    show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");
    cv::waitKey(0);
  }

  if (flag_check) {
    float* host_result;  //pointers for host memmory
    host_result = (float*)malloc(nBytes);
    memset(host_result, 0, nBytes);  //subtracting two iqual images should result in zeros
    check_result(host_ref, host_result, xsize, ysize, zsize);
    free(host_result);
  }

  //free host memory
  free(host_A);
  free(host_ref);
}
