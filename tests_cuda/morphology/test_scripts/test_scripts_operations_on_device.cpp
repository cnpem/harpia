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

//Assure that operations in gpu execute correctly for 3D data, checking with the cpu implementation
//Without breaking the input data in chunks
int test_operations_on_device() {
  std::string filenameBinary = "./example_images/binary/ILSIMG_600x1520x1520_16bits.raw";
  std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  int xsize = 600;
  int ysize = 1520;
  int zsize = 520;

  int kernel_xsize = 3;
  int kernel_ysize = 3;
  int kernel_zsize = 3;

  // Create kernel - must be a 2D kernel, because openCV opearates slice by slice
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * kernel_xsize * kernel_ysize *
                        kernel_zsize);  // Size to fit the horizontal line kernels 3x3x3
  get_structuring_element_3D(kernel, kernel_xsize, kernel_ysize, kernel_zsize);

  int ngpus = -1;   // Assures that the maximum number of gpus available  will be used

  int flag_check = 1;    // Whether to compare with OpenCV
  int flag_verbose = 0;  // Whether to print status messages
  int flag_float = 0;

  MorphChain closing = {DILATION, EROSION};
  MorphChain opening = {EROSION, DILATION};

  float memoryOccupancy = 0.9f;

  printf("\nCompare operations on device with host results in 3D\n");

  // Binary operations
  test_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                              kernel_ysize, kernel_zsize, EROSION, memoryOccupancy, ngpus, flag_check,
                              flag_verbose);

  test_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                              kernel_ysize, kernel_zsize, DILATION, memoryOccupancy, ngpus, flag_check,
                              flag_verbose);

  test_geodesic_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, EROSION,
                                       memoryOccupancy, ngpus, flag_check, flag_verbose);

  test_geodesic_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, DILATION,
                                       memoryOccupancy, ngpus, flag_check, flag_verbose);

  test_morph_chain_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                                    kernel_ysize, kernel_zsize, closing, memoryOccupancy, ngpus,
                                    flag_check, flag_verbose);

  test_morph_chain_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                                    kernel_ysize, kernel_zsize, opening, memoryOccupancy, ngpus,
                                    flag_check, flag_verbose);

  test_smooth_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, memoryOccupancy, ngpus, flag_check,
                               flag_verbose);

  test_complement_binary_on_device(filenameBinary, xsize, ysize, zsize, memoryOccupancy, ngpus, flag_check,
                                   flag_verbose);

  test_fill_holes_on_device(filenameBinary, xsize, ysize, zsize, 0, memoryOccupancy, ngpus, flag_check,
                            flag_verbose);

  // Grayscale operations
  test_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                 kernel_ysize, kernel_zsize, EROSION, memoryOccupancy, ngpus, flag_check,
                                 flag_verbose, flag_float);

  test_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                 kernel_ysize, kernel_zsize, DILATION, memoryOccupancy, ngpus, flag_check,
                                 flag_verbose, flag_float);

  test_geodesic_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, EROSION,
                                          memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float);

  test_geodesic_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, DILATION,
                                          memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float);

  test_morph_chain_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                       kernel_ysize, kernel_zsize, closing, memoryOccupancy, ngpus,
                                       flag_check, flag_verbose, flag_float);

  test_morph_chain_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                       kernel_ysize, kernel_zsize, opening, memoryOccupancy, ngpus,
                                       flag_check, flag_verbose, flag_float);

  test_subtraction_on_device(filenameGrayscale, filenameGrayscale, xsize, ysize, zsize,
                             memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float);

  test_top_hat_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                         kernel_zsize, memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float);

  test_bottom_hat_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, memoryOccupancy, ngpus, flag_check, flag_verbose, 
                            flag_float);

  test_top_hat_reconstruction_on_device(filenameGrayscale, xsize, ysize, zsize, kernel,
                                        kernel_xsize, kernel_ysize, kernel_zsize, flag_check,
                                        flag_verbose, flag_float);

  test_bottom_hat_reconstruction_on_device(filenameGrayscale, xsize, ysize, zsize, kernel,
                                           kernel_xsize, kernel_ysize, kernel_zsize, flag_check,
                                           flag_verbose, flag_float);

  free(kernel);

  return 0;
}
