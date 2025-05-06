#ifndef TEST_SCRIPTS_H
#define TEST_SCRIPTS_H

#include <string>
#include "../../morphology/morphology.h"

int test_operations_on_host();
int test_operations_on_device();
int test_chunked_executer(float memoryOccupancy);
int test_chunked_time(const std::string&csv_filename , const std::string& machineName, int ngpus, 
                      float memoryOccupancy, int repetitions, int flag_verbose);
#endif  // TEST_SCRIPTS_H