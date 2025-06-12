#include "../include/morphology/cuda_helper.h"
#include "../include/tests/morphology/test_scripts.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <string>
#include <iostream>
#include <cstdlib> // for std::atof, std::atoi


int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <gpuMemory (0 to 1)> <ngpus> <repetitions> <verbose (0 or 1)>" << std::endl;
        return 1;
    }

    float gpuMemory = std::atof(argv[1]);
    if (gpuMemory < 0.0f || gpuMemory > 1.0f) {
        std::cerr << "Error: gpuMemory must be between 0 and 1." << std::endl;
        return 1;
    }

    int ngpus = std::atoi(argv[2]);
    int repetitions = std::atoi(argv[3]);
    if (repetitions < 1) {
        std::cerr << "Error: repetitions must be greater than 0." << std::endl;
        return 1;
    }

    int flag_verbose = std::atoi(argv[4]);
    if (flag_verbose < 0 || flag_verbose > 1) {
        std::cerr << "Error: verbose must be greater than 0 (false) or 1 (true)." << std::endl;
        return 1;
    }



    std::cout << argv[0] << " Starting... " << std::endl;
    std::cout << "gpuMemory: " << gpuMemory
              << ", ngpus: " << ngpus
              << (ngpus == 0 ? " (CPU mode)"
                             : (ngpus < 0 ? " (All GPUs mode)"
                                          : " (Specific GPU count mode)"))
              << ", repetitions: " << repetitions
              << ", verbose: " << flag_verbose << std::endl;

    // test_check_device_info();
    // test_operations_on_host();
    // test_operations_on_device();
    // test_chunked_executer(gpuMemory);

    std::string machineName = "aida";

    // Encode ngpus meaning in the filename
    std::string ngpu_str;
    if (ngpus == 0)
        ngpu_str = "cpu";
    else if (ngpus < 0)
        ngpu_str = "allgpu";
    else
        ngpu_str = std::to_string(ngpus) + "gpu";

    std::string fileName = machineName + "_" + ngpu_str + "_" + std::to_string(repetitions) + "reps_test_results.csv";

    test_chunked_time(fileName, machineName, ngpus, gpuMemory, repetitions, flag_verbose);

    return 0;
}
