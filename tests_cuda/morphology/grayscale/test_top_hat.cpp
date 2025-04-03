#include <stdio.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/top_hat.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_top_hat.h"
#include "../../../include/tests/morphology/test_util.h"

void test_top_hat_on_device(const std::string& filename, const int xsize, const int ysize,
                            const int zsize, int* kernel, const int kernel_xsize,
                            const int kernel_ysize, const int kernel_zsize, float memoryOccupancy, int ngpus, 
                            const int flag_check, const int flag_verbose, const int flag_float) {

  printf("\nTest top hat on device\n");

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);

  size_t nBytes = size * sizeof(float);

  if (flag_verbose)
    printf("Matrix size:   %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *host_A, *device_ref;  //pointers for host memmory
  host_A = (float*)malloc(nBytes);
  device_ref = (float*)calloc(size, sizeof(float));

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);

  // device erosion
  int ncopies = 3;
  int operations = 2;
  chunkedExecutorKernel(top_hat_on_device<float>, ncopies, memoryOccupancy, ngpus, operations, host_A,
                        device_ref, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                        kernel_ysize, kernel_zsize);

  if (flag_check) {
    float* host_ref;
    host_ref = (float*)calloc(size, sizeof(float));

    // erosion
    top_hat_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                    kernel_zsize);

    check_result(host_ref, device_ref, xsize, ysize, zsize);

    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}

void test_top_hat_on_host(const std::string& filename, const int xsize, const int ysize,
                          const int zsize, int* kernel, const int kernel_xsize,
                          const int kernel_ysize, const int kernel_zsize, const int flag_show,
                          const int flag_check, const int flag_verbose, const int flag_float) {

  printf("\nTest top hat on host\n");

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
  if (flag_show)
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image");

  // bottomHat
  top_hat_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                  kernel_zsize);
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
    opencv_ref = (float*)calloc(size, sizeof(float));

    // opencv erosion
    morphology_3D_openCV(host_A, opencv_ref, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
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