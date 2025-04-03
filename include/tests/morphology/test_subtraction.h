#ifndef TEST_SUBTRACTION_H
#define TEST_SUBTRACTION_H

#include <string>

void test_subtraction_on_device(const std::string& filename, const std::string& filename2,
                                const int xsize, const int ysize, const int zsize,
                                float memoryOccupancy, int ngpus, const int flag_check,
                                const int flag_verbose, const int flag_float);

void test_subtraction_on_host(const std::string& filename, const std::string& filename2,
                              const int xsize, const int ysize, const int zsize,
                              const int flag_show, const int flag_check, const int flag_verbose, 
                              const int flag_float);

#endif  // TEST_SUBTRACTION_H