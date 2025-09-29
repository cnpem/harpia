#ifndef TEST_FILL_HOLES_H
#define TEST_FILL_HOLES_H

#include <string>

void test_fill_holes_on_device(const std::string& filename, const int xsize, const int ysize,
                               const int zsize, int padding, float memoryOccupancy, int ngpus, const int flag_check,
                               const int flag_verbose);

void test_fill_holes_on_host(const std::string& filename, const int xsize, const int ysize,
                             const int zsize, const int flag_verbose);

#endif  // TEST_FILL_HOLES_H