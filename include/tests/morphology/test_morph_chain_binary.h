#ifndef TEST_MORPHOLOGY_CHAIN_BINARY_H
#define TEST_MORPHOLOGY_CHAIN_BINARY_H

#include <string>
#include "../../morphology/morphology.h"

void test_morph_chain_binary_on_device(const std::string& filename, const int xsize,
                                       const int ysize, const int zsize, int* kernel,
                                       const int kernel_xsize, const int kernel_ysize,
                                       const int kernel_zsize, MorphChain chain,
                                       float memoryOccupancy, int ngpus, const int flag_check,
                                       const int flag_verbose);

void test_morph_chain_binary_on_host(const std::string& filename, const int xsize, const int ysize,
                                     const int zsize, int* kernel, const int kernel_xsize,
                                     const int kernel_ysize, const int kernel_zsize,
                                     MorphChain chain, const int flag_show, const int flag_check,
                                     const int flag_verbose);

#endif  // TEST_MORPHOLOGY_CHAIN_BINARY_H