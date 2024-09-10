#ifndef WRAPPER_H
#define WRAPPER_H

#include <stdio.h>
#include <iostream>

template <typename Func, typename dtype, typename... Args>
void wrapper(Func func, int ncopies, dtype* hostImage, dtype* hostOutput, int* kernel,
             int kernel_xsize, int kernel_ysize, int kernel_zsize, const int xsize, const int ysize,
             const int zsize, Args... args);

#include "../../src/wrapper.cu"  // Include the implementation

// template<typename Func, typename T, typename... Args>
// void wrapper(Func func, T* inputImage, T* outputImage, Args... args) {
//     // Preprocessing
//     std::cout << "Preprocessing inputImage and outputImage" << std::endl;

//     // Call the actual function with the rest of the arguments
//     func(inputImage, outputImage, args...);
// }

#endif  // WRAPPER_H