#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <string>

#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_morph_chain_grayscale.h"
#include "../../../include/tests/morphology/test_util.h"

void test_morph_chain_grayscale_on_device(const std::string& filename, const int xsize,
                                          const int ysize, const int zsize, int* kernel,
                                          const int kernel_xsize, const int kernel_ysize,
                                          const int kernel_zsize, MorphChain chain,
                                          const int flag_check, const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;

  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  //pointers for host memory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(device_ref, 0, nBytes);

  read_input(host_A, filename, size, flag_verbose);

  // device erosion
  morph_chain_grayscale_on_device(host_A, device_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                                  kernel_ysize, kernel_zsize, chain, flag_verbose);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)malloc(nBytes);
    memset(host_ref, 0, nBytes);

    // erosion
    morph_chain_grayscale_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                                  kernel_ysize, kernel_zsize, chain);

    check_result(host_ref, device_ref, xsize, ysize, zsize);

    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}

void test_morph_chain_grayscale_on_host(const std::string& filename, const int xsize,
                                        const int ysize, const int zsize, int* kernel,
                                        const int kernel_xsize, const int kernel_ysize,
                                        const int kernel_zsize, MorphChain chain,
                                        const int flag_show, const int flag_check,
                                        const int flag_verbose) {
  // set input dimension
  int size = xsize * ysize * zsize;

  size_t nBytes = size * sizeof(float);
  if (flag_verbose)
    printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *host_A, *host_ref;  //pointers for host memory
  host_A = (float*)malloc(nBytes);
  host_ref = (float*)malloc(nBytes);

  // set input data
  memset(host_A, 0, nBytes);
  memset(host_ref, 0, nBytes);
  read_input(host_A, filename, size, flag_verbose);
  if (flag_show)
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image");

  // erosion
  morph_chain_grayscale_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                                kernel_ysize, kernel_zsize, chain);
  if (flag_show)
    show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");

  if (flag_check) {
    if (kernel_zsize > 1) {
      printf(
          "WARNING: Results will not match, openCV is done slice by slice, it "
          "is incompatible with kernel zsize: %d",
          kernel_zsize);
    }
    float *opencv_ref, *opencv_tmp;
    opencv_ref = (float*)malloc(nBytes);
    opencv_tmp = (float*)malloc(nBytes);
    memset(opencv_ref, 0, nBytes);
    memset(opencv_tmp, 0, nBytes);

    // openCV erosion
    morphology_3D_openCV(host_A, opencv_tmp, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
                         chain.operation1);
    morphology_3D_openCV(opencv_tmp, opencv_ref, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
                         chain.operation2);
    if (flag_show)
      show_image_3D(opencv_ref, xsize, ysize, zsize, "Result openCV");

    check_result(host_ref, opencv_ref, xsize, ysize, zsize);

    free(opencv_ref);
    free(opencv_tmp);
  }

  if (flag_show)
    cv::waitKey(0);

  //free host memory
  free(host_A);
  free(host_ref);
}
