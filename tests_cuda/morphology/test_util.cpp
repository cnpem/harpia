#include "../../include/tests/morphology/test_util.h"
#include <sys/time.h>  // For gettimeofday
#include <cstdint>     // For uint16_t
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

/**
 * @brief Get the current CPU time in seconds.
 * 
 * This function returns the current CPU time in seconds. It uses the `gettimeofday` function
 * to get the current time and converts it to seconds.
 * 
 * @return The current CPU time in seconds.
 */
double cpu_second() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);  // Convert usec to sec and sum correctly
}

/**
 * @brief Read raw image data from file into memory.
 * 
 * This function reads raw image data from a file into a memory pointer. The data is read as
 * `uint16_t` and then converted to the specified data type.
 * 
 * @tparam dtype The data type to which the image data will be converted.
 * @param image Pointer to the memory where the image data will be stored.
 * @param filename Path to the file containing raw image data.
 * @param size Size of the data to be read in terms of number of `uint16_t` elements.
 * @param flag_verbose Flag to control verbosity of the function's output.
 */
template <typename dtype>
void read_input(dtype* image, const std::string& filename, const size_t size, const int flag_verbose, const size_t chunk_size = 1024 * 1024 * 1024) {

    // Open the raw file
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open file for reading." << std::endl;
    return;
  }

  // Buffer for reading chunks
  std::vector<uint16_t> buffer;

  // Adjust chunk size if the file is smaller
  size_t actual_chunk_size = std::min(chunk_size, size);
  buffer.resize(actual_chunk_size);

  size_t total_chunks = size / actual_chunk_size;
  size_t remaining_elements = size % actual_chunk_size;

  size_t index = 0;

  // Read and convert chunk by chunk
  for (size_t chunk = 0; chunk < total_chunks; ++chunk) {
    file.read(reinterpret_cast<char*>(buffer.data()), actual_chunk_size * sizeof(uint16_t));
    for (size_t i = 0; i < actual_chunk_size; ++i, ++index) {
      image[index] = static_cast<dtype>(buffer[i]);
    }
  }

  // Handle remaining elements
  if (remaining_elements > 0) {
    buffer.resize(remaining_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), remaining_elements * sizeof(uint16_t));
    for (size_t i = 0; i < remaining_elements; ++i, ++index) {
      image[index] = static_cast<dtype>(buffer[i]);
    }
  }

  file.close();

  if (flag_verbose) {
    std::cout << "Data has been successfully read in chunks." << std::endl;
  }
}
template void read_input<float>(float*, const std::string&, const size_t, const int, const size_t);
template void read_input<int>(int*, const std::string&, const size_t, const int, const size_t);
template void read_input<unsigned int>(unsigned int*, const std::string&, const size_t, const int, 
                                       const size_t);
template void read_input<int16_t>(int16_t*, const std::string&, const size_t, const int, 
                                  const size_t);
template void read_input<uint16_t>(uint16_t*, const std::string&, const size_t, const int, 
                                   const size_t);
template void read_input<int8_t>(int8_t*, const std::string&, const size_t, const int, 
                                 const size_t);
template void read_input<uint8_t>(uint8_t*, const std::string&, const size_t, const int, 
                                  const size_t);
/**
 * @brief Print a 3D matrix.
 * 
 * This function prints the content of a 3D matrix to the standard output. The matrix is printed
 * slice by slice, with each slice represented as a 2D matrix.
 * 
 * @tparam dtype The data type of the matrix elements.
 * @param image Pointer to the 3D matrix data.
 * @param xsize Size of the matrix in the x-dimension.
 * @param ysize Size of the matrix in the y-dimension.
 * @param zsize Size of the matrix in the z-dimension.
 */
template <typename dtype>
void show_matrix_3D(dtype* image, const int xsize, const int ysize, const int zsize) {
  dtype* im = image;
  std::cout << "\nMatrix: (" << xsize << "x" << ysize << "x" << zsize << ")\n";
  for (int idz = 0; idz < zsize; idz++) {
    std::cout << "\nslice: " << idz << "\n";
    for (int idy = 0; idy < ysize; idy++) {
      for (int idx = 0; idx < xsize; idx++) {
        std::cout << " " << im[idx];
      }
      im += xsize;
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}
template void show_matrix_3D<float>(float*, const int, const int, const int);
template void show_matrix_3D<int>(int*, const int, const int, const int);
template void show_matrix_3D<unsigned int>(unsigned int*, const int, const int, const int);
template void show_matrix_3D<int16_t>(int16_t*, const int, const int, const int);
template void show_matrix_3D<uint16_t>(uint16_t*, const int, const int, const int);
template void show_matrix_3D<int8_t>(int8_t*, const int, const int, const int);
template void show_matrix_3D<uint8_t>(uint8_t*, const int, const int, const int);

/**
 * @brief Check if two matrices match.
 * 
 * This function compares two matrices element-wise and checks if they match within a specified 
 * tolerance. It prints an error message if any element differs by more than the specified epsilon.
 * 
 * @tparam dtype The data type of the matrix elements.
 * @param test Pointer to the test matrix.
 * @param ref Pointer to the reference matrix.
 * @param nx Size of the matrix in the x-dimension.
 * @param ny Size of the matrix in the y-dimension.
 * @param nz Size of the matrix in the z-dimension.
 */
template <typename dtype>
void check_result(dtype* test, dtype* ref, const int nx, const int ny, const int nz) {

  double epsilon = 1.0E-8;  // Tolerance for floating-point comparison
  bool match = true;
  dtype* itest = test;
  dtype* iref = ref;

  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
        if (std::abs(itest[ix] - iref[ix]) > (dtype)epsilon) {
          match = false;
          std::cout << "Matrices do not match!\n";
          std::cout << "test: " << std::fixed << std::setprecision(2) << std::setw(5) << itest[ix]
                    << " ref " << std::setw(5) << iref[ix] << " at current [" << ix << "," << iy
                    << "," << iz << "]\n";
          return;  // Exit on first mismatch
        }
      }
      itest += nx;
      iref += nx;
    }
  }
  if (match)
    std::cout << "Matrices match!\n\n";
}
template void check_result<int>(int*, int*, const int, const int, const int);
template void check_result<float>(float*, float*, const int, const int, const int);