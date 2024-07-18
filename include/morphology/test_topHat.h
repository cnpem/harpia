#ifndef TEST_TOP_HAT_H
#define TEST_TOP_HAT_H

#include <string>
#include "morphology.h"

void test_topHatOnDevice(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize,
                         const int flag_check, const int flag_verbose);

void test_topHatOnHost(const std::string& filename, const int xsize, const int ysize, const int zsize,
                            int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                            const int flag_show, const int flag_check, const int flag_verbose);

#endif // TEST_TOP_HAT_H