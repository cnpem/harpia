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
                                float memoryOccupancy, int ngpus, const int flag_check,
                                const int flag_verbose, const int flag_float) {

  printf("\nTest subtraction on device\n");

  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  // set input dimension
  size_t nBytes = size * sizeof(float);

  if (flag_verbose)
    printf("Matrix size:   %zu (%d.%d.%d) \n", size, xsize, ysize, zsize);

  float *host_A, *device_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  device_ref = (float*)malloc(nBytes);

  if (host_A == nullptr || device_ref == nullptr) {
    std::cerr << "Memory allocation failed!" << std::endl;
    return;
  }

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);
  read_input(device_ref, filename2, size, flag_verbose, flag_float);

  // device erosion
  int ncopies = 2;
  chunkedExecutor(subtraction_on_device<float>, ncopies, memoryOccupancy, ngpus, host_A, device_ref, xsize,
                  ysize, zsize, flag_verbose);
  //show_image_3D(device_ref + 54 * xsize * ysize, xsize, ysize, 2, "device_ref");

  if (flag_check) {
    float* host_ref;
    host_ref = (float*)calloc(size, sizeof(float));
    read_input(host_ref, filename2, size, flag_verbose, flag_float);

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
                              const int flag_show, const int flag_check, const int flag_verbose, 
                              const int flag_float) {

  printf("\nTest subtraction on host\n");

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);

  size_t nBytes = size * sizeof(float);
  if (flag_verbose) {
    printf("Matrix size:   %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);
  }

  float *host_A, *host_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  host_ref = (float*)malloc(nBytes);

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);
  read_input(host_ref, filename, size, flag_verbose, flag_float);
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
    host_result = (float*)calloc(size, sizeof(float));
    check_result(host_ref, host_result, xsize, ysize, zsize);
    free(host_result);
  }

  //free host memory
  free(host_A);
  free(host_ref);
}
