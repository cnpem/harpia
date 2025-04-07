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
#include <chrono>
#include <fstream>
#include <iostream>
#include <functional>

template <typename Func>
size_t time_function(Func func, int repetitions) {
  if (repetitions <= 0) return 0;

  func(); // Run once to warm up GPU (not timed)

  size_t total_time = 0;
  for (int i = 0; i < repetitions; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      func();
      auto end = std::chrono::high_resolution_clock::now();

      total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

  return total_time / repetitions; // Compute mean over all measured runs
}

void log_to_csv(const std::string& filename, const std::string& machineName, int ngpus, 
  float memoryOccupancy, int repetitions, int xsize, int ysize, int zsize, 
  const std::string& dataType, const std::string& functionName, size_t execTime) {
std::ofstream file;
bool file_exists = std::ifstream(filename).good();  // Check if file already exists

file.open(filename, std::ios::app); // Append mode
if (!file_exists) {
file << "Operation,Machine,Gpus,gpuMemory,Module Time (s),Repetitions,Image Data Type,xsize,ysize,zsize\n"; 
}

file << functionName << "," << machineName << "," << ngpus << "," << memoryOccupancy << "," 
<< execTime << "," << repetitions << "," << dataType << "," << xsize << "," << ysize << "," 
<< zsize << "\n";

file.close();
}

// template <typename Func>
// size_t measure_and_log(const std::string& functionName, std::string dataType, Func test_func){
//   size_t exec_time = time_function(test_func);
//   printf("%s executed in %zu microseconds\n", functionName.c_str(), exec_time);
//   log_to_csv(csv_filename, machineName, ngpus, memoryOccupancy, repetitions, xsize, ysize, 
//   zsize, dataType, functionName, exec_time);
// };

int test_chunked_time(const std::string&csv_filename , const std::string& machineName, int ngpus, 
                      float memoryOccupancy, int repetitions) {

  // std::string filenameBinary = "./example_images/binary/Recon_2052x2052x2048_16bits.raw";
  // std::string filenameGrayscale = "./example_images/grayscale/Recon_2052x2052x2048_32bits.raw";

  std::string filenameBinary = "./example_images/binary/ILSIMG_600x1520x1520_16bits.raw";
  std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

  int xsize = 600, ysize = 1520, zsize = 1000;
  int kernel_xsize = 3, kernel_ysize = 3, kernel_zsize = 3;

  int* kernel = (int*)malloc(sizeof(int) * kernel_xsize * kernel_ysize * kernel_zsize);
  get_structuring_element_3D(kernel, kernel_xsize, kernel_ysize, kernel_zsize);

  int flag_check = 0, flag_verbose = 1, flag_float = 1;
  MorphChain closing = {DILATION, EROSION};
  MorphChain opening = {EROSION, DILATION};

  printf("\nCompare chunked operations on device with host results in 3D\n");

  // Define an inline lambda function to measure and log time
  auto measure_and_log = [&](const std::string& functionName, std::string dataType, auto test_func){
      size_t exec_time = time_function(test_func, repetitions);
      printf("%s executed in %zu microseconds\n", functionName.c_str(), exec_time);
      log_to_csv(csv_filename, machineName, ngpus, memoryOccupancy, repetitions, xsize, ysize, zsize, dataType, functionName, exec_time);
  };

  measure_and_log("complement_binary_on_device", "int16", [&]() {
      test_complement_binary_on_device(filenameBinary, xsize, ysize, zsize, memoryOccupancy, ngpus, 
                                       flag_check, flag_verbose);
  });

  measure_and_log("subtraction_on_device", "int16", [&]() {
      test_subtraction_on_device(filenameGrayscale, filenameGrayscale, xsize, ysize, zsize, 
                                 memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float);
  });

  measure_and_log("morph_binary_on_device (erosion)", "int16", [&]() {
      test_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize, 
                                  kernel_ysize, kernel_zsize, EROSION, memoryOccupancy, ngpus, 
                                  flag_check, flag_verbose);
  });

  measure_and_log("morph_binary_on_device (dilation)", "int16", [&]() {
      test_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, kernel_xsize, 
                                  kernel_ysize, kernel_zsize, DILATION, memoryOccupancy, ngpus, 
                                  flag_check, flag_verbose);
  });

  measure_and_log("morph_chain_binary_on_device (closing)", "int16", [&]() {
      test_morph_chain_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, 
                                        kernel_xsize, kernel_ysize, kernel_zsize, closing, 
                                        memoryOccupancy, ngpus, flag_check, flag_verbose);
  });

  measure_and_log("morph_chain_binary_on_device (opening)", "int16", [&]() {
      test_morph_chain_binary_on_device(filenameBinary, xsize, ysize, zsize, kernel, 
                                        kernel_xsize, kernel_ysize, kernel_zsize, opening, 
                                        memoryOccupancy, ngpus, flag_check, flag_verbose);
  });

  measure_and_log("morph_grayscale_on_device (erosion)", "int16", [&]() {
      test_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, 
                                     kernel_xsize, kernel_ysize, kernel_zsize, EROSION, 
                                     memoryOccupancy, ngpus, flag_check, flag_verbose, flag_float);
  });

  measure_and_log("morph_grayscale_on_device (dilation)", "int16", [&]() {
      test_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize, 
                                     kernel_ysize, kernel_zsize, DILATION, memoryOccupancy, ngpus, 
                                     flag_check, flag_verbose, flag_float);
  });

  measure_and_log("top_hat_on_device", "int16", [&]() {
      test_top_hat_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize, 
                             kernel_ysize, kernel_zsize, memoryOccupancy, ngpus, flag_check, 
                             flag_verbose, flag_float);
  });

  measure_and_log("bottom_hat_on_device", "int16", [&]() {
      test_bottom_hat_on_device(filenameGrayscale, xsize, ysize, zsize, kernel, kernel_xsize, 
                                kernel_ysize, kernel_zsize, memoryOccupancy, ngpus, flag_check, 
                                flag_verbose, flag_float);
  });

  measure_and_log("geodesic_morph_binary_on_device (erosion)", "int16", [&]() {
      test_geodesic_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, EROSION, 
                                           memoryOccupancy, ngpus, flag_check, flag_verbose);
  });

  measure_and_log("geodesic_morph_binary_on_device (dilation)", "int16", [&]() {
      test_geodesic_morph_binary_on_device(filenameBinary, xsize, ysize, zsize, DILATION, 
                                           memoryOccupancy, ngpus, flag_check, flag_verbose);
  });

  measure_and_log("geodesic_morph_grayscale_on_device (erosion)", "int16", [&]() {
      test_geodesic_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, EROSION, 
                                              memoryOccupancy, ngpus, flag_check, flag_verbose, 
                                              flag_float);
  });

  measure_and_log("geodesic_morph_grayscale_on_device (dilation)", "int16", [&]() {
      test_geodesic_morph_grayscale_on_device(filenameGrayscale, xsize, ysize, zsize, DILATION, 
                                              memoryOccupancy, ngpus, flag_check, flag_verbose, 
                                              flag_float);
  });

  free(kernel);
  return 0;
}
