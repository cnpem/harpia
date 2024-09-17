#include <stdio.h>
#include <cstring>
#include <string>

#include "../../../include/morphology/complement_binary.h"
#include "../../../include/tests/morphology/test_complement_binary.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_util.h"

void test_complement_binary_on_host(const std::string& filename, const int xsize, const int ysize,
                                    const int zsize, const int flag_verbose) {

  printf("\nTest binary complement on host\n");

  // set input dimension
  int size = xsize * ysize * zsize;

  size_t nBytes = size * sizeof(int);
  if (flag_verbose) {
    printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);
  }

  int *host_A, *host_ref;  //pointers for host memmory
  host_A = (int*)malloc(nBytes);
  host_ref = (int*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(host_ref, 0, nBytes);
  read_input(host_A, filename, size, flag_verbose);

  complement_binary_on_host(host_A, host_ref, size);

  show_image_3D(host_A, xsize, ysize, zsize, "Input Image");
  show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");
  cv::waitKey(0);

  //free host memory
  free(host_A);
  free(host_ref);
}

void test_complement_binary_on_device(const std::string& filename, const std::string& filename2,
                                      const int xsize, const int ysize, const int zsize,
                                      const int flag_check, const int flag_verbose) {
  int size = xsize * ysize * zsize;
  // set input dimension
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size:   %d (%d.%d.%d) \n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  //pointers for host memmory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(device_ref, 0, nBytes);

  read_input(host_A, filename, size, flag_verbose);

  // device erosion
  complement_binary_on_device(host_A, device_ref, size, flag_verbose);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)malloc(nBytes);
    memset(host_ref, 0, nBytes);

    // erosion
    complement_binary_on_host(host_A, host_ref, size);

    check_result(host_ref, device_ref, xsize, ysize, zsize);

    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}
