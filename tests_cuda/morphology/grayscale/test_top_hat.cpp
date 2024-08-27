#include <stdio.h>
#include <cstring>
#include <string>

#include "../../../include/morphology/top_hat.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_top_hat.h"
#include "../../../include/tests/morphology/test_util.h"

void test_top_hat_on_device(const std::string& filename, const int xsize, const int ysize,
                            const int zsize, int* kernel, const int kernel_xsize,
                            const int kernel_ysize, const int kernel_zsize, const int flag_check,
                            const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;

  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  //pointers for host memmory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(device_ref, 0, nBytes);

  read_input(host_A, filename, size, flag_verbose);

  // device erosion
  top_hat_on_device(host_A, device_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize,
                    ysize, zsize, flag_verbose);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)malloc(nBytes);
    memset(host_ref, 0, nBytes);

    // erosion
    top_hat_on_host(host_A, host_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize,
                    ysize, zsize, flag_verbose);

    check_result(host_ref, device_ref, xsize, ysize, zsize);

    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}

void test_top_hat_on_host(const std::string& filename, const int xsize, const int ysize,
                          const int zsize, int* kernel, const int kernel_xsize,
                          const int kernel_ysize, const int kernel_zsize, const int flag_show,
                          const int flag_check, const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;

  size_t nBytes = size * sizeof(float);
  if (flag_verbose)
    printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *host_A, *host_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  host_ref = (float*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(host_ref, 0, nBytes);
  read_input(host_A, filename, size, flag_verbose);
  if (flag_show)
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image");

  // bottomHat
  top_hat_on_host(host_A, host_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize,
                  zsize, flag_verbose);
  if (flag_show)
    show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");

  if (flag_check) {
    if (kernel_zsize > 1) {
      printf(
          "WARNING: Results will not match, opencv is done slice by slice, it "
          "is incompatible with kernel zsize: %d",
          kernel_zsize);
    }

    float* opencv_ref;
    opencv_ref = (float*)malloc(nBytes);
    memset(opencv_ref, 0, nBytes);

    // opencv erosion
    morphology_3D_openCV(host_A, opencv_ref, kernel_xsize, kernel_ysize, xsize, ysize, zsize,
                         TOPHAT);
    if (flag_show)
      show_image_3D(opencv_ref, xsize, ysize, zsize, "Result OpenCV");

    check_result(host_ref, opencv_ref, xsize, ysize, zsize);

    free(opencv_ref);
  }

  if (flag_show)
    cv::waitKey(0);

  //free host memory
  free(host_A);
  free(host_ref);
}