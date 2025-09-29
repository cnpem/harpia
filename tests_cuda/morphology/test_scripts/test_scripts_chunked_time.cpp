#include "../../../include/morphology/morphology.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/tests/morphology/test_util.h"
#include "../../../include/tests/morphology/test_scripts.h"
#include "../../../include/morphology/operations_binary.h"
#include "../../../include/morphology/operations_grayscale.h"

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


int test_chunked_time(const std::string&csv_filename , const std::string& machineName, int ngpus, 
                      float memoryOccupancy, int repetitions, int flag_verbose) {

  printf("\nMeasure chunked operations time execution on device\n");

  std::string filenameBinary = "./example_images/binary/Recon_2052x2052x2048_16bits.raw";
  std::string filenameGrayscale = "./example_images/grayscale/Recon_2052x2052x2048_32bits.raw";
  int xsize = 2052, ysize = 2052, zsize = 2048;

//   std::string filenameBinary = "./example_images/binary/ILSIMG_600x1520x1520_16bits.raw";
//   std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";
//   int xsize = 600, ysize = 1520, zsize = 1000;

  // Set input dimension
  size_t size = static_cast<size_t>(xsize) * static_cast<size_t>(ysize) * static_cast<size_t>(zsize);
  size_t nBytes = size * sizeof(int);
  int *hostImage, *deviceRef;// Pointers for host memory
  hostImage = (int*)malloc(nBytes); 
  deviceRef = (int*)calloc(size, sizeof(int));

  // Define an inline lambda function to measure and log time
  auto measure_and_log = [&](const std::string& functionName, std::string dataType, int repetitions, auto test_func){
    size_t exec_time = time_function(test_func, repetitions);
    printf("%s executed in %zu microseconds\n", functionName.c_str(), exec_time);
    log_to_csv(csv_filename, machineName, ngpus, memoryOccupancy, repetitions, xsize, ysize, zsize, dataType, functionName, exec_time);
  };

  // Create kernel
  int kernel_xsize = 3, kernel_ysize = 3, kernel_zsize = 3;
  int* kernel = (int*)malloc(sizeof(int) * kernel_xsize * kernel_ysize * kernel_zsize);
  get_structuring_element_3D(kernel, kernel_xsize, kernel_ysize, kernel_zsize);
 
  //Test binary operations--------------------------------------------------------------------------

  // Set input data
  measure_and_log("read_input", "uint16->int32", 1, [&](){
    read_input(hostImage, filenameBinary, size, flag_verbose, 0); //flag_float set to 0
  });  

  // Time tests
  measure_and_log("erosion_binary", "int32", repetitions, [&](){
    erosion_binary(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("dilation_binary", "int32", repetitions, [&](){
    dilation_binary(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                    kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("closing_binary", "int32", repetitions, [&](){
    closing_binary(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("opening_binary", "int32", repetitions, [&](){
    opening_binary(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("smooth_binary", "int32", repetitions, [&](){
    smooth_binary(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  int *hostMarker;// Pointer for host memory
  hostMarker = (int*)calloc(size, sizeof(int));

  dilation_binary(hostImage, hostMarker, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                  kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  measure_and_log("geodesic_erosion_binary", "int32", repetitions, [&](){
    geodesic_erosion_binary(hostMarker, hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, 
                            memoryOccupancy, ngpus);
  });

  erosion_binary(hostImage, hostMarker, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                 kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  measure_and_log("geodesic_dilation_binary", "int32", repetitions, [&](){
    geodesic_dilation_binary(hostMarker, hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, 
                             memoryOccupancy, ngpus);
  });

  free(hostMarker);
  hostMarker = NULL; //safety guard measure to avoid reuse it before allocating memory again;

  //Test grayscale operations-----------------------------------------------------------------------

  // Set input data
  measure_and_log("read_input", "float32->int32", 1, [&](){
    read_input(hostImage, filenameGrayscale, size, flag_verbose, 1); //flag_float set to 1
  });

  // Time tests
  measure_and_log("erosion_grayscale", "int32", repetitions, [&](){
    erosion_grayscale(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("dilation_grayscale", "int32", repetitions, [&](){
    dilation_grayscale(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                    kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("closing_grayscale", "int32", repetitions, [&](){
    closing_grayscale(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("opening_grayscale", "int32", repetitions, [&](){
    opening_grayscale(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                   kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  hostMarker = (int*)calloc(size, sizeof(int));

  dilation_grayscale(hostImage, hostMarker, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                  kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  measure_and_log("geodesic_erosion_grayscale", "int32", repetitions, [&](){
    geodesic_erosion_grayscale(hostMarker, hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, 
                               memoryOccupancy, ngpus);
  });

  erosion_grayscale(hostImage, hostMarker, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
                 kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  measure_and_log("geodesic_dilation_grayscale", "int32", repetitions, [&](){
    geodesic_dilation_grayscale(hostMarker, hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, 
                                memoryOccupancy, ngpus);
  });

  free(hostMarker);  
  //tophat e bootmhat
  measure_and_log("top_hat", "int32", repetitions, [&](){
    top_hat(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
            kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  measure_and_log("bottom_hat", "int32", repetitions, [&](){
    bottom_hat(hostImage, deviceRef, xsize, ysize, zsize, flag_verbose, kernel, kernel_xsize, 
               kernel_ysize, kernel_zsize, memoryOccupancy, ngpus);
  });

  free(hostImage);
  free(deviceRef);
  free(kernel);
  return 0;
}
