#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <string>
#include <cstring>

#include "../../../include/morphology/test_image_processing.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/test_morph_chain_binary.h"
#include "../../../include/morphology/structuring_elements.h"
#include "../../../include/morphology/test_util.h"

void test_morphChainBinaryOnDevice(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphChain chain,
                         const int flag_check, const int flag_verbose)
{
    // set input dimension
    int size = xsize*ysize*zsize;
    
    size_t nBytes = size * sizeof(int);

    if(flag_verbose) printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

    int *host_A, *device_ref; //pointers for host memmory
    host_A = (int *)malloc(nBytes);
    device_ref = (int *)malloc(nBytes);

    // set input data
    memset(host_A, 0, nBytes); 
    memset(device_ref, 0, nBytes);

    readInput(host_A,filename, size, flag_verbose);
    
    morphChainBinaryOnDevice(host_A, device_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, 
                        block_xsize, block_ysize, block_zsize, chain, flag_verbose);

    if(flag_check){
        int *host_ref;  
        host_ref = (int *)malloc(nBytes);
        memset(host_ref, 0, nBytes); 
        // erosion
        morphChainBinaryOnHost(host_A, host_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain);

        checkResult(host_ref, device_ref, xsize, ysize, zsize);
        free(host_ref);
    }

    free(host_A);
    free(device_ref);
}


void test_morphChainBinaryOnHost(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         MorphChain chain, const int flag_show, const int flag_check, const int flag_verbose)
{
    // set input dimension
    int size = xsize*ysize*zsize;

    size_t nBytes = size * sizeof(int);
    if(flag_verbose) printf("Matrix size:   %d (%d.%d.%d)\n", size, xsize, ysize, zsize);

    int *host_A, *host_ref; //pointers for host memmory
    host_A = (int *)malloc(nBytes);
    host_ref = (int *)malloc(nBytes);

    // set input data
    memset(host_A, 0, nBytes); 
    memset(host_ref, 0, nBytes);
    readInput(host_A, filename, size, flag_verbose);
    if(flag_show) showImage3D(host_A, xsize, ysize, zsize, "Input Image");

    // erosion
    morphChainBinaryOnHost(host_A, host_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, chain);
    if(flag_show) showImage3D(host_ref, xsize, ysize, zsize, "Result Image");

    if(flag_check){
        if(kernel_zsize > 1){
            printf("WARNING: Results will not match, opencv is done slice by slice, it is incompatible with kernel zsize: %d", kernel_zsize);
        } 
        int *opencv_ref, *opencv_tmp;
        opencv_ref = (int *)malloc(nBytes);
        opencv_tmp = (int *)malloc(nBytes);
        memset(opencv_ref, 0, nBytes); 
        memset(opencv_tmp, 0, nBytes); 

        // opencv erosion 
        morphology3DopenCV(host_A, opencv_tmp, kernel_xsize, kernel_ysize, xsize, ysize, zsize, chain.operation1);
        morphology3DopenCV(opencv_tmp, opencv_ref, kernel_xsize, kernel_ysize, xsize, ysize, zsize, chain.operation2);
        if(flag_show) showImage3D(opencv_ref, xsize, ysize, zsize, "Result OpenCV");

        checkResult(host_ref, opencv_ref, xsize, ysize, zsize);

        free(opencv_ref);
        free(opencv_tmp);
    }

    if(flag_show) cv::waitKey(0); // needed for the showImage3D() calls

    // free host memory
    free(host_A);
    free(host_ref);
}