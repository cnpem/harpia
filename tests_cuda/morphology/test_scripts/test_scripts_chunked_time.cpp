#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/tests/morphology/test_bottom_hat.h"
#include "../../../include/tests/morphology/test_complement_binary.h"
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>

// Time the execution on gpu for an specifc number of gpus and memoryOccupancy
// TODO: finish this implementation - time and save results in a csv.
int test_chunked_time(float memoryOccupancy, int ngpus) {

  std::string filenameBinary = "./example_images/binary/Recon_2052x2052x2048_16bits.raw";
  std::string filenameGrayscale = "./example_images/grayscale/Recon_2052x2052x2048_32bits.raw";

  int xsize = 2052;
  int ysize = 2052;
  int zsize = 2048;

  int kernel_xsize = 3;
  int kernel_ysize = 3;
  int kernel_zsize = 3;

  // Create kernel - must be a 2D kernel, because openCV opearates slice by slice
  int* kernel;
  kernel = (int*)malloc(sizeof(int) * kernel_xsize * kernel_ysize *
                        kernel_zsize);  // Size to fit the horizontal line kernels 3x3x3
  get_structuring_element_3D(kernel, kernel_xsize, kernel_ysize, kernel_zsize);

  int flag_check = 0;    // Whether to compare with host. In this test it is not needed to compare with host.
  int flag_verbose = 1;  // Whether to print status messages
  int flag_float = 1;

  MorphChain closing = {DILATION, EROSION};
  MorphChain opening = {EROSION, DILATION};

  printf("\nCompare chunked operations on device with host results in 3D\n");

  // ChunkedExecutor
  test_complement_binary_on_device(filenameBinary, xsize, ysize, zsize, memoryOccupancy, ngpus, flag_check,
                                   flag_verbose); 

  test_subtraction_on_device(filenameGrayscale, filenameGrayscale, xsize, ysize, zsize,
                             memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float); 

  // ChunkedExecutorKernel
  test_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, EROSION, memoryOccupancy, ngpus, flag_check,
                               flag_verbose); 

  test_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, DILATION, memoryOccupancy, ngpus, flag_check,
                               flag_verbose); 

  test_morph_chain_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                                     kernel_ysize, kernel_zsize, closing, memoryOccupancy, ngpus,
                                     flag_check, flag_verbose);

  test_morph_chain_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                                     kernel_ysize, kernel_zsize, opening, memoryOccupancy, ngpus,
                                     flag_check, flag_verbose); 

  test_smooth_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize,
                               kernel_ysize, kernel_zsize, memoryOccupancy, ngpus, flag_check,
                               flag_verbose); 

  test_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                 kernel_ysize, kernel_zsize, EROSION, memoryOccupancy, ngpus, flag_check,
                                 flag_verbose, flag_float); 

  test_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                 kernel_ysize, kernel_zsize, DILATION, memoryOccupancy, ngpus, flag_check,
                                 flag_verbose, flag_float); 

  test_morph_chain_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                       kernel_ysize, kernel_zsize, closing, memoryOccupancy, ngpus,
                                       flag_check, flag_verbose, flag_float); 

  test_morph_chain_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                                       kernel_ysize, kernel_zsize, opening, memoryOccupancy, ngpus,
                                       flag_check, flag_verbose, flag_float); 

  test_top_hat_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize,
                         kernel_zsize, memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float); 

  test_bottom_hat_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize,
                            kernel_ysize, kernel_zsize, memoryOccupancy, ngpus, flag_check, flag_verbose, 
                            flag_float); 

  // ChunkedExecutorGeodesic
  test_geodesic_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, EROSION,
                                       memoryOccupancy, ngpus, flag_check, flag_verbose); 

  test_geodesic_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, DILATION,
                                       memoryOccupancy, ngpus, flag_check, flag_verbose); 

  test_geodesic_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, EROSION,
                                          memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float); 

  test_geodesic_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, DILATION,
                                          memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float); 

  free(kernel);

  return 0;
}
