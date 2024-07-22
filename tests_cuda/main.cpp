#include "../include/morphology/test_scripts.h"
#include "../include/morphology/cuda_helper.h"


#include<stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char **argv){

    // // printf("%s Starting... \n", argv[0]);

    test_check_device_info();

    // printf("######################################################\n");
    // printf("# TEST 1: basic morphological operations consistency #\n");
    // printf("######################################################\n\n");
    // test_script1(EROSION);
    // test_script1(DILATION);

    // printf("#########################################################\n");
    // printf("# TEST 2: basic morphological operations execution time #\n");
    // printf("#########################################################\n\n");
    // test_script2();

    // printf("#############################################\n");
    // printf("# TEST 2: combined morphological operations #\n");
    // printf("#############################################\n\n");
    // MorphChain opening = {EROSION, DILATION};
    // MorphChain closing = {DILATION, EROSION};

    // test_script3(opening);
    // test_script3(closing);
    

    // test_script4();
    test_script5();

    return 0;
}