#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/tests/morphology/test_bottom_hat.h"
#include "../../../include/tests/morphology/test_bottom_hat_reconstruction.h"
#include "../../../include/tests/morphology/test_complement_binary.h"
#include "../../../include/tests/morphology/test_fill_holes.h"
#include "../../../include/tests/morphology/test_geodesic_morph_binary.h"
#include "../../../include/tests/morphology/test_geodesic_morph_grayscale.h"
#include "../../../include/tests/morphology/test_morph_binary.h"
#include "../../../include/tests/morphology/test_morph_chain_binary.h"
#include "../../../include/tests/morphology/test_morph_chain_grayscale.h"
#include "../../../include/tests/morphology/test_morph_grayscale.h"
#include "../../../include/tests/morphology/test_scripts.h"
#include "../../../include/tests/morphology/test_smooth_binary.h"
#include "../../../include/tests/morphology/test_subtraction.h"
#include "../../../include/tests/morphology/test_top_hat.h"
#include "../../../include/tests/morphology/test_top_hat_reconstruction.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>

// Assure that operations check with opencv implementation or make visual sense according to the task
// Tests of correctness - these tests are important to validate the operations on c/c++.
// It is important to have a cpu trustworth implementation to check if cuda operations are correct
int test_operations_on_host() {
  std::string filenameBinary = "./example_images/binary/blobs_355x321x1_16b.raw";
  std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  // Create kernel - must be a 2D kernel, because openCV opearates slice by slice
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * 25);  // Size to fit the horizontal line kernels 3x3x3
  get_structuring_element_3D(kernel, 5, 5, 1);

  int flag_show = 1;     // Whether to plot the result
  int flag_check = 1;    // Whether to compare with OpenCV
  int flag_verbose = 1;  // Whether to print status messages
  int flag_float = 0;

  MorphChain closing = {DILATION, EROSION};
  MorphChain opening = {EROSION, DILATION};

  printf("\nCompare operations on host results with OpenCV in 2D\n");

  // Binary
  test_morph_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, EROSION, flag_show,
                            flag_check, flag_verbose);

  test_morph_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, DILATION, flag_show,
                            flag_check, flag_verbose);

  test_morph_chain_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, closing, flag_show,
                                  flag_check, flag_verbose);

  test_morph_chain_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, opening, flag_show,
                                  flag_check, flag_verbose);

  test_smooth_binary_on_host(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, flag_show, flag_check,
                             flag_verbose);

  // Grayscale
  test_morph_grayscale_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, EROSION, flag_show,
                               flag_check, flag_verbose, flag_float);

  test_morph_grayscale_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, DILATION,
                               flag_show, flag_check, flag_verbose, flag_float);

  test_morph_chain_grayscale_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, closing,
                                     flag_show, flag_check, flag_verbose, flag_float);

  test_morph_chain_grayscale_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, opening,
                                     flag_show, flag_check, flag_verbose, flag_float);

  test_subtraction_on_host(filenameGrayscale, filenameGrayscale, 600, 1520, 1, flag_show,
                           flag_check, flag_verbose, flag_float);

  test_top_hat_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, flag_show, flag_check,
                       flag_verbose, flag_float);

  test_bottom_hat_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, flag_show, flag_check,
                          flag_verbose, flag_float);

  // Visual tests
  printf("\nVisualy evaluate operations on host in 2D\n");

  test_geodesic_morph_grayscale_on_host(filenameGrayscale, 600, 1520, 1, EROSION, flag_verbose, 
                                        flag_float);

  test_geodesic_morph_grayscale_on_host(filenameGrayscale, 600, 1520, 1, DILATION, flag_verbose, 
                                        flag_float);

  test_top_hat_reconstruction_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1,
                                      flag_verbose, flag_float);


  test_bottom_hat_reconstruction_on_host(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1,
                                         flag_verbose, flag_float);

  test_geodesic_morph_binary_on_host(filenameBinary, 355, 321, 1, EROSION, flag_verbose);

  test_geodesic_morph_binary_on_host(filenameBinary, 355, 321, 1, DILATION, flag_verbose);

  test_complement_binary_on_host(filenameBinary, 355, 321, 1, flag_verbose);

  filenameBinary = "./example_images/binary/eagle_275x183x1_16b.raw";
  test_fill_holes_on_host(filenameBinary, 275, 183, 1, flag_verbose);

  free(kernel);

  return 0;
}
