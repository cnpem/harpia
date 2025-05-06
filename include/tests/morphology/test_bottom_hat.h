#ifndef TEST_BOTTOM_HAT_H
#define TEST_BOTTOM_HAT_H

#include <string>
#include "../../morphology/morphology.h"

void test_bottom_hat_on_device(const std::string& filename, const int xsize, const int ysize,
                               const int zsize, int* kernel, const int kernel_xsize,
                               const int kernel_ysize, const int kernel_zsize,
                               float memoryOccupancy, int ngpus, const int flag_check, const int flag_verbose, 
                               const int flag_float);

void test_bottom_hat_on_host(const std::string& filename, const int xsize, const int ysize,
                             const int zsize, int* kernel, const int kernel_xsize,
                             const int kernel_ysize, const int kernel_zsize, const int flag_show,
                             const int flag_check, const int flag_verbose, 
                             const int flag_float);

#endif  // TEST_BOTTOM_HAT_H