#ifndef TEST_GEODESIC_MORPHOLOGY_BINARY_H
#define TEST_GEODESIC_MORPHOLOGY_BINARY_H

#include <string>
#include "../../morphology/morphology.h"

void test_geodesic_morph_binary_on_device(const std::string& filename, const int xsize,
                                          const int ysize, const int zsize, MorphOp operation,
                                          float memoryOccupancy, int ngpus, const int flag_check,
                                          const int flag_verbose);

void test_geodesic_morph_binary_on_host(const std::string& filename, const int xsize,
                                        const int ysize, const int zsize, MorphOp operation,
                                        const int flag_verbose);

#endif  // TEST_GEODESIC_MORPHOLOGY_BINARY_H