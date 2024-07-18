#ifndef TEST_MORPHOLOGY_GRAYSCALE_H
#define TEST_MORPHOLOGY_GRAYSCALE_H

#include <string>
#include "morphology.h"

void test_morphGrayscaleOnDevice(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphOp operation,
                         const int flag_check, const int flag_verbose);

void test_morphGrayscaleOnHost(const std::string& filename, const int xsize, const int ysize, const int zsize,
                            int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize, 
                            MorphOp operation, const int flag_show, const int flag_check, const int flag_verbose);

void test_morphGrayscaleOnDeviceTime(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphOp operation, int n);

void test_morphGrayscaleTimeCompare(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphOp operation);

#endif // TEST_MORPHOLOGY_GRAYSCALE_H