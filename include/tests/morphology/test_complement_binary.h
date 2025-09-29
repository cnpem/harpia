#ifndef TEST_COMPLEMENT_BINARY_H
#define TEST_COMPLEMENT_BINARY_H

#include <string>

void test_complement_binary_on_device(const std::string& filename, const int xsize, const int ysize,
                                      const int zsize, float memoryOccupancy, int ngpus, const int flag_check,
                                      const int flag_verbose);

void test_complement_binary_on_host(const std::string& filename, const int xsize, const int ysize,
                                    const int zsize, const int flag_verbose);

#endif  // TEST_COMPLEMENT_BINARY_H