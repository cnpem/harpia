#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <iostream>
#include <string>

// Function prototypes
double cpu_second();

template <typename dtype>
void read_input(dtype* image, const std::string& filename, const size_t size, 
                const int flag_verbose, const int flag_float = 0);

template <typename dtype>
void show_matrix_3D(dtype* image, const int xsize, const int ysize, const int zsize);

template <typename dtype>
void check_result(dtype* testRef, dtype* opencvRef, const int nx, const int ny, const int nz);

#endif  // TEST_UTIL_H