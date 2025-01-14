#include "../include/morphology/cuda_helper.h"
#include "../include/tests/morphology/test_scripts.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <iostream>
#include <cstdlib> // For atof

int main(int argc, char** argv) {
    // Check if the user provided the gpuMemory parameter
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gpuMemory (0 to 1)>" << std::endl;
        return 1; // Exit with error code
    }

    // Parse the gpuMemory parameter
    float gpuMemory = std::atof(argv[1]);

    // Validate that gpuMemory is in the range [0, 1]
    if (gpuMemory < 0.0f || gpuMemory > 1.0f) {
        std::cerr << "Error: gpuMemory must be between 0 and 1." << std::endl;
        return 1; // Exit with error code
    }

    std::cout << argv[0] << " Starting... " << std::endl;
    std::cout << "gpuMemory parameter: " << gpuMemory << std::endl;

    test_check_device_info();
    // test_operations_on_host();
    // test_operations_on_device();
    test_chunked_executer(gpuMemory);

    return 0;
}