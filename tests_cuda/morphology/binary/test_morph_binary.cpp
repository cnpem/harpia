#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <cstring>
#include <string>

#include "../../../include/common/chunkedExecutor.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/tests/morphology/test_image_processing.h"
#include "../../../include/tests/morphology/test_morph_binary.h"
#include "../../../include/tests/morphology/test_util.h"

/**
 * @brief Tests the binary morphological operations on the GPU.
 * 
 * @param filename The name of the input file.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the structuring element kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param operation The morphological operation to perform (e.g., erosion, dilation).
 * @param flag_check If set, the results will be checked for correctness.
 * @param flag_verbose If set, additional information will be printed.
 */
void test_morph_binary_on_device(const std::string& filename, const int xsize, const int ysize,
                                 const int zsize, int* kernel, const int kernel_xsize,
                                 const int kernel_ysize, const int kernel_zsize, MorphOp operation,
                                 float memoryOccupancy, int ngpus, const int flag_check,
                                 const int flag_verbose) {

  printf("\nTest binary %s on device\n", (operation ? "dilation" : "erosion"));

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);

  int *host_A, *device_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes); 
  device_ref = (int*)calloc(size, sizeof(int));

  // Set input data
  read_input(host_A, filename, size, flag_verbose);

  int ncopies = 2;
  int operations = 1;
  chunkedExecutorKernel(morph_binary_on_device<int>, ncopies, memoryOccupancy, ngpus, operations, host_A,
                        device_ref, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize,
                        kernel_ysize, kernel_zsize, operation);

  if (flag_check) {
    int* host_ref;
    host_ref = (int*)calloc(size, sizeof(int));
    // Perform binary morphology on host for comparison
    morph_binary_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                         kernel_zsize, operation);
    check_result(host_ref, device_ref, xsize, ysize, zsize);
    free(host_ref);
  }

  free(host_A);
  free(device_ref);
}

/**
 * @brief Tests the custom binary erosion operation and compares its result with OpenCV's 
 * implementation.
 * 
 * @param filename The name of the input file.
 * @param xsize The size of the image in the x-dimension.
 * @param ysize The size of the image in the y-dimension.
 * @param zsize The size of the image in the z-dimension.
 * @param kernel Pointer to the structuring element kernel.
 * @param kernel_xsize The size of the kernel in the x-dimension.
 * @param kernel_ysize The size of the kernel in the y-dimension.
 * @param kernel_zsize The size of the kernel in the z-dimension.
 * @param operation The morphological operation to perform (e.g., erosion, dilation).
 * @param flag_show If set, the input and result images will be displayed.
 * @param flag_check If set, the results will be checked for correctness.
 * @param flag_verbose If set, additional information will be printed.
 */
void test_morph_binary_on_host(const std::string& filename, const int xsize, const int ysize,
                               const int zsize, int* kernel, const int kernel_xsize,
                               const int kernel_ysize, const int kernel_zsize, MorphOp operation,
                               const int flag_show, const int flag_check, const int flag_verbose) {

  printf("\nTest binary %s on host\n", (operation ? "dilation" : "erosion"));

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);

  if (flag_verbose)
    printf("Matrix size: %zu (%d.%d.%d)\n", size, xsize, ysize, zsize);


  int *host_A, *host_ref;  // Pointers for host memory
  host_A = (int*)malloc(nBytes);
  host_ref = (int*)calloc(size, sizeof(int));

  // Set input data
  read_input(host_A, filename, size, flag_verbose);
  if (flag_show)
    show_image_3D(host_A, xsize, ysize, zsize, "Input Image");
  
  // Perform binary morphology on host
  morph_binary_on_host(host_A, host_ref, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                       kernel_zsize, operation);

  if (flag_show)
    show_image_3D(host_ref, xsize, ysize, zsize, "Result Image");

  if (flag_check) {
    if (kernel_zsize > 1) {
      printf(
          "WARNING: Results will not match, OpenCV is done slice by slice, it "
          "is incompatible with kernel zsize: %d",
          kernel_zsize);
    }

    int* opencv_ref;
    opencv_ref = (int*)calloc(size, sizeof(int));

    // Perform OpenCV erosion
    morphology_3D_openCV(host_A, opencv_ref, xsize, ysize, zsize, kernel_xsize, kernel_ysize,
                         operation);
    if (flag_show)
      show_image_3D(opencv_ref, xsize, ysize, zsize, "Result OpenCV");

    check_result(host_ref, opencv_ref, xsize, ysize, zsize);

    free(opencv_ref);
  }

  if (flag_show)
    cv::waitKey(0);  // Needed for the show_image_3D() calls

  // Free host memory
  free(host_A);
  free(host_ref);
}

// /**
//  * @brief Measures the time taken to perform binary morphological operations on the GPU.
//  *
//  * @param filename The name of the input file.
//  * @param xsize The size of the image in the x-dimension.
//  * @param ysize The size of the image in the y-dimension.
//  * @param zsize The size of the image in the z-dimension.
//  * @param kernel Pointer to the structuring element kernel.
//  * @param kernel_xsize The size of the kernel in the x-dimension.
//  * @param kernel_ysize The size of the kernel in the y-dimension.
//  * @param kernel_zsize The size of the kernel in the z-dimension.
//  * @param operation The morphological operation to perform (e.g., erosion, dilation).
//  * @param n The number of iterations to average the time over.
//  */
// void test_morph_binary_on_device_time(const std::string& filename, const int xsize, const int ysize,
//                                       const int zsize, int* kernel, const int kernel_xsize,
//                                       const int kernel_ysize, const int kernel_zsize,
//                                       MorphOp operation, int n) {

//   int flag_check = 0;
//   int flag_verbose = 0;

//   double iStart, iElaps;
//   iElaps = 0;

//   for (int i = 0; i < n; i++) {
//     iStart = cpu_second();
//     test_morph_binary_on_device(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
//                                 kernel_zsize, operation, flag_check, flag_verbose);
//     iElaps += cpu_second() - iStart;
//   }
//   iElaps /= n;
//   printf("\nmorph_binary_on_device Mean time elapsed %f sec\n", iElaps);
// }

// /**
//  * @brief Compares the execution time of binary morphological operations on the GPU and CPU.
//  *
//  * @param filename The name of the input file.
//  * @param xsize The size of the image in the x-dimension.
//  * @param ysize The size of the image in the y-dimension.
//  * @param zsize The size of the image in the z-dimension.
//  * @param kernel Pointer to the structuring element kernel.
//  * @param kernel_xsize The size of the kernel in the x-dimension.
//  * @param kernel_ysize The size of the kernel in the y-dimension.
//  * @param kernel_zsize The size of the kernel in the z-dimension.
//  * @param operation The morphological operation to perform (e.g., erosion, dilation).
//  */
// void test_morph_binary_time_compare(const std::string& filename, const int xsize, const int ysize,
//                                     const int zsize, int* kernel, const int kernel_xsize,
//                                     const int kernel_ysize, const int kernel_zsize,
//                                     MorphOp operation) {
//   int flag_show = 0;
//   int flag_check = 0;
//   int flag_verbose = 0;

//   double iStart, iElapsHostBinary, iElapsDeviceBinary;
//   iElapsHostBinary = 0;
//   iElapsDeviceBinary = 0;

//   iStart = cpu_second();
//   test_morph_binary_on_device(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
//                               kernel_zsize, operation, flag_check, flag_verbose);
//   iElapsDeviceBinary = cpu_second() - iStart;
//   printf("\nmorph_binary_on_device Time elapsed %f sec\n", iElapsDeviceBinary);

//   iStart = cpu_second();
//   test_morph_binary_on_host(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
//                             kernel_zsize, operation, flag_show, flag_check, flag_verbose);
//   iElapsHostBinary = cpu_second() - iStart;
//   printf("\nmorph_binary_on_host Time elapsed %f sec\n", iElapsHostBinary);
// }