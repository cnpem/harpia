#include "../include/morphology/cuda_helper.h"
#include "../include/tests/morphology/test_scripts.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char** argv) {

  printf("%s Starting... \n", argv[0]);

  test_check_device_info();
  test_operations_on_host();
  test_operations_on_device();
  test_chunked_executer();

  return 0;
}