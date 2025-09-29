#include <stdio.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/complement_binary.h"
#include "../../../include/tests/morphology/test_complement_binary.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_util.h"

void test_complement_binary_on_device(const std::string& filename, const int xsize, const int ysize,
                                      const int zsize, float memoryOccupancy, int ngpus, 
                                      const int flag_check, const int flag_verbose) {

  printf("\nTest binary complement on device\n");

  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  // set input dimension
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size:   %zu (%d.%d.%d) \n", size, xsize, ysize, zsize);
  
  int *host_A, *device_ref;  //pointers for host memmory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)calloc(size, sizeof(int));

  if (host_A == nullptr || device_ref == nullptr) {
      std::cerr << "Memory allocation failed!" << std::endl;
      return;
  }

  // set input data
  read_input(host_A, filename, size, flag_verbose);

  // device erosion
  int ncopies = 2;
  chunkedExecutor(complement_binary_on_device<int>, ncopies, memoryOccupancy, ngpus, host_A, device_ref,
                  xsize, ysize, zsize, flag_verbose);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)calloc(size, sizeof(int));

    // erosion
    complement_binary_on_host(host_A, host_ref, size);

    check_result(host_ref, device_ref, xsize, ysize, zsize);

    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}

void test_complement_binary_on_host(const std::string& filename, const int xsize, const int ysize,
                                    const int zsize, const int flag_verbose) {

  printf("\nTest binary complement on host\n");

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);

  size_t nBytes = size * sizeof(int);
  if (flag_verbose) {
    printf("Matrix size:   %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);
  }

  int *host_A, *host_ref;  //pointers for host memmory

  host_A = (int*)malloc(nBytes);
  host_ref = (int*)calloc(size, sizeof(int));

  // set input data
  read_input(host_A, filename, size, flag_verbose);

  complement_binary_on_host(host_A, host_ref, size);

  show_image_3D(host_A, xsize, ysize, zsize, "Input Image");
  show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");
  cv::waitKey(0);

  //free host memory
  free(host_A);
  free(host_ref);
}
