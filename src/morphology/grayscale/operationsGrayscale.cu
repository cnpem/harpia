#include "../../../include/morphology/morphGrayscale.h"
#include "../../../include/morphology/morphChainGrayscale.h"
#include "../../../include/morphology/operationsGrayscale.h"

template<typename dtype>
void erosionGrayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                 const int flag_verbose)
{
    morphGrayscaleOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, 
                        EROSION, flag_verbose);
}
template void erosionGrayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, 
                                const int);
template void erosionGrayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);
template void erosionGrayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);


template<typename dtype>
void dilationGrayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                 const int flag_verbose)
{
    morphGrayscaleOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, 
                        DILATION, flag_verbose);
}
template void dilationGrayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, 
                                    const int);
template void dilationGrayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);
template void dilationGrayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);


template<typename dtype>
void closingGrayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                        const int flag_verbose){
   
   MorphChain closing = {DILATION, EROSION};

   morphChainGrayscaleOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                            block_xsize, block_ysize, block_zsize, closing, flag_verbose);
}
template void closingGrayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                            const int, const int);
template void closingGrayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);
template void closingGrayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);

template<typename dtype>
void openingGrayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                        const int flag_verbose){
   
   MorphChain opening = {EROSION, DILATION};

   morphChainGrayscaleOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                            block_xsize, block_ysize, block_zsize, opening, flag_verbose);
}
template void openingGrayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                            const int, const int);
template void openingGrayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);
template void openingGrayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);

 