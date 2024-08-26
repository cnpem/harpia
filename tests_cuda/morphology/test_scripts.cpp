#include "../../include/morphology/cuda_helper.h"
#include "../../include/morphology/morphology.h"
#include "../../include/morphology/structuring_elements.h"
#include "../../include/tests/morphology/test_bottom_hat.h"
#include "../../include/tests/morphology/test_fill_holes.h"
#include "../../include/tests/morphology/test_image_processing.h"
#include "../../include/tests/morphology/test_morph_binary.h"
#include "../../include/tests/morphology/test_morph_chain_binary.h"
#include "../../include/tests/morphology/test_morph_chain_grayscale.h"
#include "../../include/tests/morphology/test_morph_grayscale.h"
#include "../../include/tests/morphology/test_subtraction.h"
#include "../../include/tests/morphology/test_top_hat.h"
#include "../../include/tests/morphology/test_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>
//#include <test_bottomHat.h>

/**
 * @brief Test erosion/dilation operations for 2D/3D, binary/grayscale images.
 * 
 * This function performs morphological operations such as erosion and dilation on binary
 * and grayscale images. It tests both host and device implementations and compares results.
 * 
 * @param operation Specifies the morphological operation to test: 
 *                  EROSION or DILATION.
 * 
 * @return Returns 0 on success.
 */
int test_script1(MorphOp operation) {
  std::string filenameBinary =
      "./example_images/binary/blobs_355x321x1_16b.raw";
  std::string filenameGrayscale =
      "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  // Create kernel
  int* kernel;
  kernel = (int*)malloc(sizeof(int) *
                        27);  // Size to fit the horizontal line kernels 3x3x3
  get_structuring_element_3D(kernel, 5, 5, 1);

  int flag_show = 1;     // Whether to plot the result
  int flag_check = 1;    // Whether to compare with OpenCV erosion
  int flag_verbose = 0;  // Whether to print status messages

  printf("\n1 - Compare erosion results with OpenCV.\n");

  printf("\nTest binary %s in 2D.\n", (operation ? "dilation" : "erosion"));
  test_morph_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1,
                            operation, flag_show, flag_check, flag_verbose);

  printf("\nTest grayscale %s in 2D.\n", (operation ? "dilation" : "erosion"));
  test_morph_grayscale_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1,
                               operation, flag_show, flag_check, flag_verbose);

  printf("\n2 - Compare device and host %s.\n",
         (operation ? "dilation" : "erosion"));
  // 25x faster on device than on host

  // test_check_device_info();

  // Be careful with the maximum number of threads per block!
  // block_xsize*block_ysize*block_zsize < 1024 for the notebook GPU

  printf("\nTest cuda binary %s.\n", (operation ? "dilation" : "erosion"));
  test_morph_binary_on_device(filenameBinary, 355, 321, 1, kernel, 5, 5, 1,
                              operation, flag_check, flag_verbose);

  printf("\nTest cuda grayscale %s.\n", (operation ? "dilation" : "erosion"));
  test_morph_grayscale_on_device(filenameGrayscale, 600, 1520, 1, kernel, 5, 5,
                                 1, operation, flag_check, flag_verbose);

  flag_show = 1;
  flag_check = 0;
  flag_verbose = 0;

  // It is not possible to check the host erosion with OpenCV for different kernels because the check function
  // was developed to use only rectangular kernels of 1's

  printf("\n4 - Test horizontal line kernel on binary %s.\n",
         (operation ? "dilation" : "erosion"));
  horizontal_line_kernel(kernel);
  show_matrix_3D(kernel, 3, 3, 1);
  test_morph_binary_on_host(filenameBinary, 355, 321, 1, kernel, 3, 3, 1,
                            operation, flag_show, flag_check, flag_verbose);

  printf("\n5 - Test vertical line kernel on binary %s.\n\n",
         (operation ? "dilation" : "erosion"));
  vertical_line_kernel(kernel);
  show_matrix_3D(kernel, 3, 3, 1);
  test_morph_binary_on_host(filenameBinary, 355, 321, 1, kernel, 3, 3, 1,
                            operation, flag_show, flag_check, flag_verbose);

  // You can't test line kernels on grayscale operations because the grayscale operation only distinguishes care and don't care pixels.
  // Background and foreground values (0's and 1's) on the kernel are treated the same way.

  free(kernel);

  return 0;
}

/**
 * @brief Compare execution time of erosion/dilation on host/device.
 * 
 * This function compares the execution time of erosion/dilation operations on the host and the device.
 * 
 * @return Returns 0 on success.
 */
int test_script2() {
  // Using a binary file, it is possible to perform either the binary erosion or the grayscale operation
  std::string filename =
      "./example_images/binary/ILSIMG_600x1520x1520_16bits.raw";

  // Create kernel
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * 27);
  get_structuring_element_3D(kernel, 3, 3, 3);

  // test_check_device_info();

  // Be careful with the maximum number of threads per block!
  // block_xsize * block_ysize * block_zsize < 1024 for the notebook GPU

  int xsize = 600;
  int ysize = 1520;
  int zsize = 500;  // Maximum size to execute on laptop

  int kernel_xsize = 3;
  int kernel_ysize = 3;
  int kernel_zsize = 3;

  test_morph_binary_time_compare(filename, xsize, ysize, zsize, kernel,
                                 kernel_xsize, kernel_ysize, kernel_zsize,
                                 EROSION);

  test_morph_grayscale_time_compare(filename, xsize, ysize, zsize, kernel,
                                    kernel_xsize, kernel_ysize, kernel_zsize,
                                    EROSION);

  test_morph_grayscale_on_device_time(filename, xsize, ysize, zsize, kernel,
                                      kernel_xsize, kernel_ysize, kernel_zsize,
                                      EROSION, 10);

  test_morph_binary_on_device_time(filename, xsize, ysize, zsize, kernel,
                                   kernel_xsize, kernel_ysize, kernel_zsize,
                                   EROSION, 10);

  return 0;
}

/**
 * @brief Test opening/closing operations.
 * 
 * This function tests morphological opening and closing operations on binary and grayscale images.
 * It compares the results between host and device implementations.
 * 
 * @param chain Specifies the morphological chain of operations to test.
 * 
 * @return Returns 0 on success.
 */
int test_script3(MorphChain chain) {
  std::string filenameBinary =
      "./example_images/binary/blobs_355x321x1_16b.raw";
  std::string filenameGrayscale =
      "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  MorphChain closing = {DILATION, EROSION};
  const int closing_flag = (chain.operation1 == closing.operation1) &&
                           (chain.operation2 == closing.operation2);

  // Create kernel
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * 125);
  get_structuring_element_3D(kernel, 5, 5, 5);

  int flag_show = 1;     // Whether to plot the result
  int flag_check = 1;    // Whether to compare with OpenCV erosion
  int flag_verbose = 0;  // Whether to print status messages

  printf("\n1 - Compare erosion results with OpenCV.\n");

  printf("\nTest binary %s in 2D.\n", (closing_flag ? "closing" : "opening"));
  test_morph_chain_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1,
                                  chain, flag_show, flag_check, flag_verbose);

  printf("\nTest grayscale %s in 2D.\n",
         (closing_flag ? "closing" : "opening"));
  test_morph_chain_grayscale_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5,
                                     5, 1, chain, flag_show, flag_check,
                                     flag_verbose);

  printf("\n2 - Compare device and host %s.\n",
         (closing_flag ? "dilation" : "erosion"));
  // 25x faster on device than on host

  // test_check_device_info();

  // Be careful with the maximum number of threads per block!
  // block_xsize * block_ysize * block_zsize < 1024 for the notebook GPU

  printf("\nTest cuda binary %s.\n", (closing_flag ? "closing" : "opening"));
  test_morph_chain_binary_on_device(filenameBinary, 355, 321, 1, kernel, 5, 5,
                                    1, chain, flag_check, flag_verbose);

  printf("\nTest cuda grayscale %s.\n", (closing_flag ? "closing" : "opening"));
  test_morph_chain_grayscale_on_device(filenameGrayscale, 600, 1520, 10, kernel,
                                       5, 5, 5, chain, flag_check,
                                       flag_verbose);

  free(kernel);

  return 0;
}

/**
 * @brief Test subtraction operations.
 * 
 * This function tests the subtraction operation on grayscale images using both host and device implementations.
 * 
 * @return Returns 0 on success.
 */
int test_script4() {
  std::string filenameGrayscale =
      "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  int flag_show = 1;     // Whether to plot the result
  int flag_check = 1;    // Whether to compare with OpenCV erosion
  int flag_verbose = 1;  // Whether to print status messages

  printf("\nTest subtraction on host.\n");
  test_subtraction_on_host(filenameGrayscale, filenameGrayscale, 600, 1520, 2,
                           flag_show, flag_verbose);

  printf("\nTest subtraction on device.\n");
  test_subtraction_on_device(filenameGrayscale, filenameGrayscale, 600, 1520,
                             500, flag_check, flag_verbose);

  return 0;
}

/**
 * @brief Test topHat and bottomHat transformations.
 * 
 * This function tests the topHat and bottomHat transformations on grayscale images using both host and device implementations.
 * 
 * @return Returns 0 on success.
 */
int test_script5() {
  std::string filenameGrayscale =
      "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  // Create kernel
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * 125);
  get_structuring_element_3D(kernel, 5, 5, 5);

  int flag_show = 1;     // Whether to plot the result
  int flag_check = 0;    // Whether to compare with OpenCV erosion
  int flag_verbose = 0;  // Whether to print status messages

  printf("\n1 - Compare topHat and bottomHat results with OpenCV.\n");

  printf("\nTest grayscale bottomHat in 2D.\n");
  test_bottom_hat_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1,
                          flag_show, flag_check, flag_verbose);

  printf("\nTest grayscale topHat in 2D.\n");
  test_top_hat_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1,
                       flag_show, flag_check, flag_verbose);

  printf("\n2 - Compare device and host.\n");
  flag_verbose = 1;
  printf("\nTest grayscale bottomHat in 3D.\n");
  test_bottom_hat_on_device(filenameGrayscale, 600, 1520, 100, kernel, 5, 5, 5,
                            flag_check, flag_verbose);

  printf("\nTest grayscale topHat in 3D.\n");
  test_top_hat_on_device(filenameGrayscale, 600, 1520, 100, kernel, 5, 5, 5,
                         flag_check, flag_verbose);

  free(kernel);

  return 0;
}

int test_script6() {
  //Atention: if th einput image has a white border (of 1's), the algorithm 'breaks'
  // The fill_holes F image(marker/input) will be completely 0, this way the reconstruction by dilation
  // has no start point, the output will be completely zero.

  std::string filenameBinary =
      "./example_images/binary/eagle_275x183x1_16b.raw";

  int flag_show = 1;     // Whether to plot the result
  int flag_check = 1;    // Whether to compare with OpenCV erosion
  int flag_verbose = 0;  // Whether to print status messages

  test_fill_holes_on_host(filenameBinary, 275, 183, 1, flag_show, 0,
                          flag_verbose);

  test_fill_holes_on_device(filenameBinary, 355, 321, 1, flag_check,
                            flag_verbose);

  return 0;
}