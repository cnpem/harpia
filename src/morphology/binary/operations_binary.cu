#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/operations_binary.h"

template<typename dtype>
void erosionBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                 const int flag_verbose)
{
    morphBinaryOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, 
                        EROSION, flag_verbose);
}
template void erosionBinary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, 
                                const int);
template void erosionBinary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);
template void erosionBinary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);


template<typename dtype>
void dilationBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                 const int flag_verbose)
{
    morphBinaryOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, 
                        DILATION, flag_verbose);
}
template void dilationBinary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, const int, 
                                    const int);
template void dilationBinary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);
template void dilationBinary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                        const int, const int);


template<typename dtype>
void closingBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                        const int flag_verbose){
   
   MorphChain closing = {DILATION, EROSION};

   morphChainBinaryOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                            block_xsize, block_ysize, block_zsize, closing, flag_verbose);
}
template void closingBinary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                            const int, const int);
template void closingBinary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);
template void closingBinary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);


template<typename dtype>
void openingBinary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int block_xsize, const int block_ysize, const int block_zsize, 
                        const int flag_verbose){
   
   MorphChain opening = {EROSION, DILATION};

   morphChainBinaryOnDevice(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                            block_xsize, block_ysize, block_zsize, opening, flag_verbose);
}
template void openingBinary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int, const int, 
                                            const int, const int);
template void openingBinary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);
template void openingBinary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, 
                                                  const int, const int, const int, const int);

 