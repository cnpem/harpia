#ifndef TEST_GEODESIC_MORPHOLOGY_GRAYSCALE_H
#define TEST_GEODESIC_MORPHOLOGY_GRAYSCALE_H

#include <string>
#include "../../morphology/morphology.h"

void test_geodesic_morph_grayscale_on_device(const std::string& filename, const int xsize,
                                             const int ysize, const int zsize, MorphOp operation,
                                             float memoryOccupancy, int ngpus, const int flag_check,
                                             const int flag_verbose, const int flag_float);

void test_geodesic_morph_grayscale_on_host(const std::string& filename, const int xsize,
                                           const int ysize, const int zsize, MorphOp operation,
                                           const int flag_verbose, const int flag_float);

#endif  // TEST_GEODESIC_MORPHOLOGY_GRAYSCALE_H