#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
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
                                       float memoryOccupancy, int ngpus, const int flag_check,
                                       const int flag_verbose) {

  const int closing_flag = (chain.operation1 == DILATION) && (chain.operation2 == EROSION);
  printf("\nTest binary %s on device\n", (closing_flag ? "closing" : "opening"));

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  device_ref = (int*)calloc(size, sizeof(int));

  // Read input data from file
  read_input(host_A, filename, size, flag_verbose);

  int ncopies = 3;
  int operations = 2;
  chunkedExecutorKernel(morph_chain_binary_on_device<int>, ncopies, memoryOccupancy, ngpus, operations,
                        host_A, device_ref, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                        kernel_ysize, kernel_zsize, chain);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)calloc(size, sizeof(int));

    morph_chain_binary_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, chain);
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

  const int closing_flag = (chain.operation1 == DILATION) && (chain.operation2 == EROSION);
  printf("\nTest binary %s on host\n", (closing_flag ? "closing" : "opening"));

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *host_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  host_ref = (int*)calloc(size, sizeof(int));

  // Initialize memory
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
    opencv_ref = (int*)calloc(size, sizeof(int));
    opencv_tmp = (int*)malloc(nBytes);

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
