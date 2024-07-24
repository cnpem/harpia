#include <stdio.h>
#include <string>
#include <cstring>

#include "../../../include/morphology/test_subtraction.h"
#include "../../../include/morphology/subtraction.h"
#include "../../../include/morphology/test_image_processing.h"
#include "../../../include/morphology/test_util.h"

void test_subtractionOnDevice(const std::string& filename, const std::string& filename2, const int xsize, const int ysize, const int zsize,
                              const int flag_check, const int flag_verbose)
{
    int size = xsize*ysize*zsize;
    // set input dimension
    size_t nBytes = size * sizeof(int);

    if(flag_verbose) printf("Matrix size:   %d (%d.%d.%d) \n", size, xsize, ysize, zsize);

    int *host_A, *host_B, *device_ref; //pointers for host memmory
    host_B = (int *)malloc(nBytes);
    host_A = (int *)malloc(nBytes);
    device_ref = (int *)malloc(nBytes);

    // set input data
    memset(host_B, 0, nBytes); 
    memset(host_A, 0, nBytes); 
    memset(device_ref, 0, nBytes);

    readInput(host_A,filename, size, flag_verbose);
    readInput(host_B,filename2, size, flag_verbose);

    // device erosion 
    subtraction(host_A, host_B, device_ref, size, flag_verbose);

    if(flag_check){
        int *host_ref;
        host_ref = (int *)malloc(nBytes);
        memset(host_ref, 0, nBytes); 

        // erosion
        subtractionOnHost(host_A, host_B, host_ref, size);

        checkResult(host_ref, device_ref, xsize, ysize, zsize);

        free(host_ref);
    }

    free(host_B);
    free(host_A);
    free(device_ref);
}


void test_subtractionOnHost(const std::string& filename, const std::string& filename2, const int xsize, const int ysize, const int zsize,
                            const int flag_show, const int flag_verbose)
{
    // set input dimension
    int size = xsize*ysize*zsize;

    size_t nBytes = size * sizeof(float);
    if(flag_verbose) printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

    float *host_A, *host_B, *host_ref; //pointers for host memmory
    host_A = (float *)malloc(nBytes);
    host_B = (float *)malloc(nBytes);
    host_ref = (float *)malloc(nBytes);

    // set input data
    memset(host_A, 1, nBytes); 
    memset(host_B, 1, nBytes); 
    memset(host_ref, 1, nBytes); 
    readInput(host_A, filename, size, flag_verbose);
    readInput(host_B, filename, size, flag_verbose);
    if(flag_show) showImage3D(host_A, xsize, ysize, zsize, "Input Image A");
    if(flag_show) showImage3D(host_B, xsize, ysize, zsize, "Input Image B");

    // bottomHat
    subtractionOnHost(host_A, host_B, host_ref, size);
    if(flag_show) showImage3D(host_ref, xsize, ysize, zsize, "Result Image");

    if(flag_show) cv::waitKey(0);

    //free host memory
    free(host_A);
    free(host_B);
    free(host_ref);
}