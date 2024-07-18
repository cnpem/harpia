#include "../../include/morphology/morphology.h"
#include "../../include/morphology/test_util.h"
#include "../../include/morphology/test_morph_binary.h"
#include "../../include/morphology/test_morph_grayscale.h"
#include "../../include/morphology/test_morph_chain_binary.h"
#include "../../include/morphology/test_morph_chain_grayscale.h"
#include "../../include/morphology/cuda_helper.h"
#include "../../include/morphology/structuring_elements.h"
#include "../../include/morphology/test_top_hat.h"
#include "../../include/morphology/test_bottom_hat.h"
#include "../../include/morphology/test_image_processing.h"
#include "../../include/morphology/test_subtraction.h"

#include<stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>
//#include <test_bottomHat.h>

// Test erosion/dilation for 2D/3D, binary/grayscale images
int test_script1(MorphOp operation){

    std::string filenameBinary = "./example_images/binary/blobs_355x321x1_16b.raw";
    std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";


    // create kernel   
    int *kernel;
    kernel = (int*)malloc(sizeof(int)*27); 
    getStructuringElement3D(kernel, 5, 5, 1);

    int flag_show = 1; //whether to plot the result
    int flag_check = 1; //whether to compare with opencv erosion
    int flag_verbose = 0; //whether to print sattus messages

    printf("\n1 - Compare erosion results with OpenCV.\n");

    printf("\nTest binary %s in 2D.\n", (operation ? "dilation" : "erosion"));
    test_morphBinaryOnHost(filenameBinary, 355, 321, 1, kernel, 5, 5, 1,operation, flag_show, flag_check, flag_verbose);
    
    printf("\nTest grayscale %s in 2D.\n", (operation ? "dilation" : "erosion"));
    test_morphGrayscaleOnHost(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1,operation, flag_show, flag_check, flag_verbose);

    printf("\n2 - Compare device and host %s.\n", (operation ? "dilation" : "erosion"));
    //25x mais rápido no device do que no host

    //test_check_device_info();

    // Be careful with the maximum number of threads per block! 
    // block_xsize*block_ysize*block_zsize < 1024 for the notebook GPU 

    printf("\nTest cuda binary %s.\n", (operation ? "dilation" : "erosion"));
    test_morphBinaryOnDevice(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, 32, 32, 1, operation, flag_check, flag_verbose);

    printf("\nTest cuda grayscale %s.\n", (operation ? "dilation" : "erosion"));
    test_morphGrayscaleOnDevice(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, 32, 32, 1, operation, flag_check, flag_verbose);

    flag_show = 1; 
    flag_check = 0;
    flag_verbose=0;

    // it is not possible to check the host erosion with opencv for different kernels, because the check function
    // was developed to use only rectangular kernels of 1's

    printf("\n4 - Test horizontal line kernel on binary %s.\n", (operation ? "dilation" : "erosion"));
    horizontal_line_kernel(kernel);
    showMatrix3D(kernel, 3, 3, 1);
    test_morphBinaryOnHost(filenameBinary, 355, 321, 1, kernel, 3, 3, 1, operation, flag_show, flag_check, flag_verbose);
    
    printf("\n5 - Test vertical line kernel on binary %s.\n\n", (operation ? "dilation" : "erosion"));
    vertical_line_kernel(kernel);
    showMatrix3D(kernel, 3, 3, 1);
    test_morphBinaryOnHost(filenameBinary, 355, 321, 1, kernel, 3, 3, 1, operation, flag_show, flag_check, flag_verbose);

    // you can't test line kernels on grayscale operation, because the grayscale operation only distinguishe care and don't care pixels. 
    // background and foregrousd values (0's and 1's) on the kernel are treated the same way.

    free(kernel);

    return 0;
}

// Compare execution time of erosion/dilation on host/device
int test_script2(){

    //Using a binary file it is possible to perform eather the binary erosion or the grayscale operation
    std::string filename = "./example_images/binary/ILSIMG_600x1520x1520_16bits.raw";

   // create kernel   
    int *kernel;
    kernel = (int*)malloc(sizeof(int)*27); 
    getStructuringElement3D(kernel, 3, 3, 3);

    //test_check_device_info();

    // Be careful with the maximum number of threads per block! 
    // block_xsize*block_ysize*block_zsize < 1024 for the notebook GPU 

    int xsize=600; 
    int ysize=1520; 
    int zsize=500; //maximum size to execute on laptop

    int kernel_xsize=3; 
    int kernel_ysize=3; 
    int kernel_zsize=3;

    int block_xsize = 8; 
    int block_ysize = 8; 
    int block_zsize = 8;

    test_morphBinaryTimeCompare(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                               block_xsize, block_ysize, block_zsize, EROSION);

    test_morphGrayscaleTimeCompare(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                               block_xsize, block_ysize, block_zsize, EROSION);
                               
    test_morphGrayscaleOnDeviceTime(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                               block_xsize, block_ysize, block_zsize, EROSION, 10);

    test_morphBinaryOnDeviceTime(filename, xsize, ysize, zsize, kernel, kernel_xsize, kernel_ysize, kernel_zsize,
                               block_xsize, block_ysize, block_zsize, EROSION, 10);

    return 0;
}

//Test opening/closing
int test_script3(MorphChain chain){

    std::string filenameBinary = "./example_images/binary/blobs_355x321x1_16b.raw";
    std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

    MorphChain closing = {DILATION, EROSION};
    const int closing_flag = (chain.operation1 == closing.operation1)&&(chain.operation2 == closing.operation2);

    // create kernel   
    int *kernel;
    kernel = (int*)malloc(sizeof(int)*27); 
    getStructuringElement3D(kernel, 5, 5, 5);

    int flag_show = 1; //whether to plot the result
    int flag_check = 1; //whether to compare with opencv erosion
    int flag_verbose = 0; //whether to print sattus messages

    printf("\n1 - Compare erosion results with OpenCV.\n");

    printf("\nTest binary %s in 2D.\n", (closing_flag ? "closing" : "opening"));
    test_morphChainBinaryOnHost(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, chain, flag_show, flag_check, flag_verbose);
    
    printf("\nTest grayscale %s in 2D.\n", (closing_flag ? "closing" : "opening"));
    test_morphChainGrayscaleOnHost(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, chain, flag_show, flag_check, flag_verbose);

    printf("\n2 - Compare device and host %s.\n", (closing_flag ? "dilation" : "erosion"));
    //25x mais rápido no device do que no host

    //test_check_device_info();

    // Be careful with the maximum number of threads per block! 
    // block_xsize*block_ysize*block_zsize < 1024 for the notebook GPU 

    printf("\nTest cuda binary %s.\n", (closing_flag ? "closing" : "opening"));
    test_morphChainBinaryOnDevice(filenameBinary, 355, 321, 1, kernel, 5, 5, 1, 32, 32, 1, chain, flag_check, flag_verbose);

    printf("\nTest cuda grayscale %s.\n", (closing_flag ? "closing" : "opening"));
    test_morphChainGrayscaleOnDevice(filenameGrayscale, 600, 1520, 10, kernel, 5, 5, 5, 32, 32, 1, chain, flag_check, flag_verbose);

    free(kernel);

    return 0;
}
//Test subtraction
int test_script4(){

    std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

    int flag_show = 1; //whether to plot the result
    int flag_check = 1; //whether to compare with opencv erosion
    int flag_verbose = 1; //whether to print sattus messages

    printf("\nTest subtraction on host.\n");
    test_subtractionOnHost(filenameGrayscale, filenameGrayscale, 600, 1520, 2, flag_show, flag_verbose);

    int block_size = 8;
    printf("\nTest subtraction on device.\n");
    test_subtractionOnDevice(filenameGrayscale, filenameGrayscale, 600, 1520, 500, block_size, flag_check, flag_verbose);

    return 0;
}

//Test topHat BottomHat
int test_script5(){

    std::string filenameGrayscale = "./example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw";

    // create kernel   
    int *kernel;
    kernel = (int*)malloc(sizeof(int)*27); 
    getStructuringElement3D(kernel, 5, 5, 1);

    int flag_show = 1; //whether to plot the result
    int flag_check = 0; //whether to compare with opencv erosion
    int flag_verbose = 0; //whether to print sattus messages

    printf("\n1 - Compare topHat and bottomHat results with OpenCV.\n");

    printf("\nTest grayscale bottomHat in 2D.\n");
    test_bottomHatOnHost(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, flag_show, flag_check, flag_verbose);

    printf("\nTest grayscale topHat in 2D.\n");
    test_topHatOnHost(filenameGrayscale, 600, 1520, 1, kernel, 5, 5, 1, flag_show, flag_check, flag_verbose);

    printf("\n2 - Compare device and host.\n");
    flag_verbose = 1;
    printf("\nTest grayscale bottomHat in 3D.\n");
    test_bottomHatOnDevice(filenameGrayscale, 600, 1520, 100, kernel, 5, 5, 5, 8,8,8, flag_check, flag_verbose);

    printf("\nTest grayscale topHat in 3D.\n");
    test_topHatOnDevice(filenameGrayscale, 600, 1520, 100, kernel, 5, 5, 5, 8,8,8, flag_check, flag_verbose);
   
    free(kernel);

    return 0;
}