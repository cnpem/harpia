#include <stdlib.h>
#include <cstring>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_morph_grayscale.h"
#include "../../../include/tests/morphology/test_util.h"

void test_morph_grayscale_on_device(const std::string& filename, const int xsize, const int ysize,
                                    const int zsize, int* kernel, const int kernel_xsize,
                                    const int kernel_ysize, const int kernel_zsize,
                                    MorphOp operation, float memoryOccupancy, int ngpus, const int flag_check,
                                    const int flag_verbose, const int flag_float) {

  printf("\nTest grayscale %s on device\n", (operation ? "dilation" : "erosion"));

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;

  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size:   %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  //pointers for host memory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)calloc(size, sizeof(int));

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);

  int ncopies = 2;
  int operations = 1;
  chunkedExecutorKernel(morph_grayscale_on_device<int>, ncopies, memoryOccupancy, ngpus, operations,
                        host_A, device_ref, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                        kernel_ysize, kernel_zsize, operation);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)calloc(size, sizeof(int));

    // erosion
    morph_grayscale_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, operation);

    check_result(host_ref, device_ref, xsize, ysize, zsize);

    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}
void test_morph_grayscale_on_host(const std::string& filename, const int xsize, const int ysize,
                                  const int zsize, int* kernel, const int kernel_xsize,
                                  const int kernel_ysize, const int kernel_zsize, MorphOp operation,
                                  const int flag_show, const int flag_check,
                                  const int flag_verbose, const int flag_float) {

  printf("\nTest grayscale %s on host\n", (operation ? "dilation" : "erosion"));

  // set input dimension
  size_t size = static_cast<size_t>(xsize) * ysize * zsize;
  printf("check 0\n");
  size_t nBytes = size * sizeof(float);
  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  float *host_A, *host_ref;  //pointers for host memory
  host_A = (float*)malloc(nBytes);
  host_ref = (float*)calloc(size, sizeof(float));
  printf("check 1\n");

  // set input data
  read_input(host_A, filename, size, flag_verbose, flag_float);
  if (flag_show) {
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image");
  }
  printf("check 2\n");

  // operation on host
  morph_grayscale_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                          kernel_zsize, operation);
  if (flag_show) {
    show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");
  }
  printf("check 3\n");

  if (flag_check) {
    if (kernel_zsize > 1) {
      printf(
          "WARNING: Results will not match, openCV is done slice by slice, it "
          "is incompatible with kernel zsize: %d",
          kernel_zsize);
    }

    float* opencv_ref;
    opencv_ref = (float*)calloc(size, sizeof(float));
    printf("check 4\n");

    // openCV operation
    morphology_3D_openCV(host_A, opencv_ref, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
                         operation);
    if (flag_show) {
      show_image_3D(opencv_ref, xsize, ysize, zsize, "Result openCV");
    }
    printf("check 5\n");

    check_result(host_ref, opencv_ref, xsize, ysize, zsize);
    printf("check 6\n");

    free(opencv_ref);
  }

  if (flag_show) {
    cv::waitKey(0);
  }

  //free host memory
  free(host_A);
  free(host_ref);
}

// void test_morph_grayscale_on_device_time(const std::string& filename, const int xsize,
//                                          const int ysize, const int zsize, int* kernel,
//                                          const int kernel_xsize, const int kernel_ysize,
//                                          const int kernel_zsize, MorphOp operation, int n) {
//   int flag_check = 0;
//   int flag_verbose = 0;

//   double iStart, iElaps;
//   iElaps = 0;

//   for (int i = 0; i < n; i++) {
//     iStart = cpu_second();
//     test_morph_grayscale_on_device(filename, xsize, ysize, zsize, kernel, kernel_xsize,
//                                    kernel_ysize, kernel_zsize, operation, flag_check, flag_verbose);
//     iElaps += cpu_second() - iStart;
//   }
//   iElaps = iElaps / n;
//   printf("\nmorph_grayscale_on_device Mean time elapsed %f sec\n", iElaps);
// }

// void test_morph_grayscale_time_compare(const std::string& filename, const int xsize,
//                                        const int ysize, const int zsize, int* kernel,
//                                        const int kernel_xsize, const int kernel_ysize,
//                                        const int kernel_zsize, MorphOp operation) {
//   int flag_show = 0;
//   int flag_check = 0;
//   int flag_verbose = 0;

//   double iStart, iElapsHostGrayscale, iElapsDeviceGrayscale;
//   iElapsHostGrayscale = 0;
//   iElapsDeviceGrayscale = 0;

//   iStart = cpu_second();
//   test_morph_grayscale_on_device(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
//                                  kernel_zsize, operation, flag_check, flag_verbose);
//   iElapsDeviceGrayscale = cpu_second() - iStart;
//   printf("\nmorph_grayscale_on_device Time elapsed %f sec\n", iElapsDeviceGrayscale);

//   iStart = cpu_second();
//   test_morph_grayscale_on_host(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
//                                kernel_zsize, operation, flag_show, flag_check, flag_verbose);
//   iElapsHostGrayscale = cpu_second() - iStart;
//   printf("\nmorph_grayscale_on_host Time elapsed %f sec\n", iElapsHostGrayscale);
// }
