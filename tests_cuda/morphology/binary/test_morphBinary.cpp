#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <string>
#include <cstring>

#include "../../../include/morphology/morphBinary.h"
#include "../../../include/morphology/test_morphBinary.h"
#include "../../../include/morphology/test_util.h"
#include "../../../include/morphology/test_imageProcessing.h"

void test_morphBinaryOnDevice(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphOp operation,
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
    
    morphBinaryOnDevice(host_A, device_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, 
                        block_xsize, block_ysize, block_zsize, operation, flag_verbose);

    if(flag_check){
        int *host_ref;  
        host_ref = (int *)malloc(nBytes);
        memset(host_ref, 0, nBytes); 
        // erosion
        morphBinaryOnHost(host_A, host_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, operation);

        checkResult(host_ref, device_ref, xsize, ysize, zsize);
        free(host_ref);
    }

    free(host_A);
    free(device_ref);
}

//This code executes a custum made binary erosion operation and compares its result with opencv implemented erosion
void test_morphBinaryOnHost(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         MorphOp operation, const int flag_show, const int flag_check, const int flag_verbose)
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
    morphBinaryOnHost(host_A, host_ref, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, operation);
    if(flag_show) showImage3D(host_ref, xsize, ysize, zsize, "Result Image");

    if(flag_check){
        if(kernel_zsize > 1){
            printf("WARNING: Results will not match, opencv is done slice by slice, it is incompatible with kernel zsize: %d", kernel_zsize);
        } 

        int *opencv_ref;
        opencv_ref = (int *)malloc(nBytes);
        memset(opencv_ref, 0, nBytes); 

        // opencv erosion 
        morphology3DopenCV(host_A, opencv_ref, kernel_xsize, kernel_ysize, xsize, ysize, zsize, operation);
        if(flag_show) showImage3D(opencv_ref, xsize, ysize, zsize, "Result OpenCV");

        checkResult(host_ref, opencv_ref, xsize, ysize, zsize);

        free(opencv_ref);
    }

    if(flag_show) cv::waitKey(0); // needed for the showImage3D() calls

    // free host memory
    free(host_A);
    free(host_ref);
}

void test_morphBinaryOnDeviceTime(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphOp operation, int n)
{
    int flag_check=0;
    int flag_verbose=0;

    double iStart, iElaps;
    iElaps = 0;
    
    for(int i = 0; i < n; i++){
        iStart = cpuSecond();
        test_morphBinaryOnDevice(filename, xsize, ysize, zsize, kernel,  
                                kernel_xsize, kernel_ysize, kernel_zsize, 
                                block_xsize, block_ysize, block_zsize, 
                                operation, flag_check, flag_verbose);
        iElaps += cpuSecond() - iStart;
    }
    iElaps = iElaps/n;
    printf("\n morphBinaryOnDevice Mean time elapsed %f sec\n", iElaps);

}

void test_morphBinaryTimeCompare(const std::string& filename, const int xsize, const int ysize, const int zsize,
                         int *kernel, const int kernel_xsize, const int kernel_ysize, const int kernel_zsize,
                         const int block_xsize, const int block_ysize, const int block_zsize, MorphOp operation)
{
    int flag_show=0;
    int flag_check=0;
    int flag_verbose=0;

    double iStart, iElapsHostBinary, iElapsDeviceBinary;
    iElapsHostBinary = 0;
    iElapsDeviceBinary = 0;

    iStart = cpuSecond();
    test_morphBinaryOnDevice(filename, xsize, ysize, zsize, kernel,  
                             kernel_xsize, kernel_ysize, kernel_zsize, 
                             block_xsize, block_ysize, block_zsize, 
                             operation, flag_check, flag_verbose);
    iElapsDeviceBinary = cpuSecond() - iStart;
    printf("\n morphBinaryOnDevice Time elapsed %f sec\n", iElapsDeviceBinary);

    iStart = cpuSecond();
    test_morphBinaryOnHost(filename, xsize, ysize, zsize, 
                                kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                                operation, flag_show, flag_check, flag_verbose);
    iElapsHostBinary = cpuSecond() - iStart;
    printf("\n morphBinaryOnHost Time elapsed %f sec\n", iElapsHostBinary);

}