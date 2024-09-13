#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <string>

#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_morph_chain_binary.h"
#include "../../../include/tests/morphology/test_util.h"

/**
 * @brief Tests the binary morphological operations performed in a chain on the GPU.
 * 
 * @param filename The name of the input file containing image data.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the structuring element kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param chain The chain of morphological operations to be applied.
 * @param flag_check If set, the results will be compared against host results for correctness.
 * @param flag_verbose If set, additional information about the processing will be printed.
 */
void test_morph_chain_binary_on_device(const std::string& filename, const int xsize,
                                       const int ysize, const int zsize, int* kernel,
                                       const int kernel_xsize, const int kernel_ysize,
                                       const int kernel_zsize, MorphChain chain,
                                       const int flag_check, const int flag_verbose) {
  // Set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)malloc(nBytes);

  // Initialize memory
  memset(host_A, 0, nBytes);
  memset(device_ref, 0, nBytes);

  // Read input data from file
  read_input(host_A, filename, size, flag_verbose);

  // Apply morphological chain operations on the device (GPU)
  morph_chain_binary_on_device(host_A, device_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, chain, flag_verbose);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)malloc(nBytes);
    memset(host_ref, 0, nBytes);

    // Apply morphological chain operations on the host (CPU) for comparison
    morph_chain_binary_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, chain);

    // Check results for correctness
    check_result(host_ref, device_ref, xsize, ysize, zsize);
    free(host_ref);
  }

  // Free allocated memory
  free(host_A);
  free(device_ref);
}

/**
 * @brief Tests the binary morphological operations performed in a chain on the CPU.
 * 
 * @param filename The name of the input file containing image data.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the structuring element kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param chain The chain of morphological operations to be applied.
 * @param flag_show If set, the input and result images will be displayed.
 * @param flag_check If set, the results will be compared against OpenCV's results for correctness.
 * @param flag_verbose If set, additional information about the processing will be printed.
 */
void test_morph_chain_binary_on_host(const std::string& filename, const int xsize, const int ysize,
                                     const int zsize, int* kernel, const int kernel_xsize,
                                     const int kernel_ysize, const int kernel_zsize,
                                     MorphChain chain, const int flag_show, const int flag_check,
                                     const int flag_verbose) {
  // Set input dimension
  int size = xsize * ysize * zsize;
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *host_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  host_ref = (int*)malloc(nBytes);

  // Initialize memory
  memset(host_A, 0, nBytes);
  memset(host_ref, 0, nBytes);
  read_input(host_A, filename, size, flag_verbose);
  if (flag_show)
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image");

  // Apply morphological chain operations on the host (CPU)
  morph_chain_binary_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                             kernel_ysize, kernel_zsize, chain);
  if (flag_show)
    show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");

  if (flag_check) {
    if (kernel_zsize > 1) {
      printf(
          "WARNING: Results may not match, OpenCV processes slice by slice, "
          "which may be incompatible with kernel zsize: %d",
          kernel_zsize);
    }
    int *opencv_ref, *opencv_tmp;
    opencv_ref = (int*)malloc(nBytes);
    opencv_tmp = (int*)malloc(nBytes);
    memset(opencv_ref, 0, nBytes);
    memset(opencv_tmp, 0, nBytes);

    // Apply OpenCV erosion in chain
    morphology_3D_openCV(host_A, opencv_tmp, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
                         chain.operation1);
    morphology_3D_openCV(opencv_tmp, opencv_ref, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
                         chain.operation2);
    if (flag_show)
      show_image_3D(opencv_ref, xsize, ysize, zsize, "Result OpenCV");

    // Check results for correctness
    check_result(host_ref, opencv_ref, xsize, ysize, zsize);

    free(opencv_ref);
    free(opencv_tmp);
  }

  if (flag_show)
    cv::waitKey(0);  // Needed for the show_image_3D() calls

  // Free allocated memory
  free(host_A);
  free(host_ref);
}
