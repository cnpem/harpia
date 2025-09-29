#include <stdio.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/bottom_hat_reconstruction.h"
#include "../../../include/tests/morphology/test_bottom_hat_reconstruction.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_util.h"

// Testing bottom_hat avizo also
void test_bottom_hat_reconstruction_on_host(const std::string& filename, const int xsize,
                                            const int ysize, const int zsize, int* kernel,
                                            const int kernel_xsize, const int kernel_ysize,
                                            const int kernel_zsize, const int flag_verbose, 
                                            const int flag_float) {

  printf("\nTest Avizo's bottom hat on host\n");

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);

  size_t nBytes = size * sizeof(float);
  if (flag_verbose)
    printf("Matrix size:   %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *host_A, *host_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  host_ref = (float*)calloc(size, sizeof(float));

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);

  bottom_hat_reconstruction_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                                    kernel_ysize, kernel_zsize);

  show_image_3D(host_A, xsize, ysize, 1, "Input");
  show_image_3D(host_ref, xsize, ysize, 1, "bottomHat Avizo");
  cv::waitKey(0);

  //free host memory
  free(host_A);
  free(host_ref);
}

void test_bottom_hat_reconstruction_on_device(const std::string& filename, const int xsize,
                                              const int ysize, const int zsize, int* kernel,
                                              const int kernel_xsize, const int kernel_ysize,
                                              const int kernel_zsize, const int flag_check,
                                              const int flag_verbose, const int flag_float) {

  printf("\nTest Avizo's bottom hat on device\n");

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);

  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size:   %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  //pointers for host memmory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)calloc(size, sizeof(int));

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);

  bottom_hat_reconstruction_on_device(host_A, device_ref, xsize, ysize, zsize, flag_verbose, kernel,
                                      kernel_xsize, kernel_ysize, kernel_zsize);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)calloc(size, sizeof(int));

    bottom_hat_reconstruction_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                                      kernel_ysize, kernel_zsize);

    check_result(host_ref, device_ref, xsize, ysize, zsize);
    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}
