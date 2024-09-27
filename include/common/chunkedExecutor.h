#ifndef CHUNKED_EXECUTOR_H
#define CHUNKED_EXECUTOR_H

#include <stdio.h>
#include <iostream>

template <typename Func, typename dtype, typename... Args>
void chunkedExecutor(Func func, int ncopies, const float safetyMargin, dtype* image, dtype* output,
                     const int xsize, const int ysize, const int zsize, const int verbose,
                     Args... args);

template <typename Func, typename dtype, typename... Args>
void chunkedExecutorKernel(Func func, int ncopies, const float safetyMargin, dtype* image,
                           dtype* output, const int xsize, const int ysize, const int zsize,
                           int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize,
                           const int verbose, Args... args);

// Include the implementation to avoid compilation linkage errors
// (this is the same as defining the funcition in the header file)
#include "../../src/chunkedExecutor.cu"

#endif  // CHUNKED_EXECUTOR_H