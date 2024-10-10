#include "../include/morphology/cuda_helper.h"
#include "../include/tests/morphology/test_scripts.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char** argv) {

  printf("%s Starting... \n", argv[0]);

  // test_check_device_info();
  // test_operations_on_host();
  // test_operations_on_device();
  test_chunked_executer();

  // printf("######################################################\n");
  // printf("# TEST 1: basic morphological operations consistency #\n");
  // printf("######################################################\n\n");
  // test_script1(EROSION);
  // test_script1(DILATION);

  // printf("#########################################################\n");
  // printf("# TEST 2: basic morphological operations execution time #\n");
  // printf("#########################################################\n\n");
  // // test_script2();

  // printf("#############################################\n");
  // printf("# TEST 3: combined morphological operations #\n");
  // printf("#############################################\n\n");
  // MorphChain opening = {EROSION, DILATION};
  // MorphChain closing = {DILATION, EROSION};

  // test_script3(opening);
  // test_script3(closing);

  // printf("#############################################\n");
  // printf("# TEST 4: subtraction #\n");
  // printf("#############################################\n\n");
  // test_script4();

  // printf("#############################################\n");
  // printf("# TEST 5: topHat and bottomHat #\n");
  // printf("#############################################\n\n");
  // test_script5();

  // printf("#############################################\n");
  // printf("# TEST 6: fillHoles #\n");
  // printf("#############################################\n\n");
  // test_script6();

  // test_script7();

  return 0;
}