#ifndef TEST_MORPHOLOGY_CHAIN_GRAYSCALE_H
#define TEST_MORPHOLOGY_CHAIN_GRAYSCALE_H

#include <string>
#include "morphology.h"

void test_morphChainBinaryOnDevice(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         MorphChain chain, const int flag_check, const int flag_verbose);

void test_morphChainGrayscaleOnHost(const std::string& filename, const int xsize, const int ysize, const int zsize,
                            int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize, 
                            MorphChain chain, const int flag_show, const int flag_check, const int flag_verbose);

#endif // TEST_MORPHOLOGY_CHAIN_GRAYSCALE_H