#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <iostream>
#include <string>

// Function prototypes
double cpuSecond();

template<typename dtype>
void readInput(dtype *image, const std::string& filename, const int size, const int flag_verbose);

template<typename dtype>
void showMatrix3D(dtype *image, const int xsize, const int ysize, const int zsize);

template<typename dtype>
void checkResult(dtype *testRef, dtype *opencvRef, const int nx, const int ny, const int nz);

#endif // TEST_UTIL_H