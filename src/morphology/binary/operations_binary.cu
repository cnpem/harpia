#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/operations_binary.h"

template<typename dtype>
void erosion_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    morph_binary_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, EROSION, flag_verbose);
}
template void erosion_binary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void erosion_binary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void erosion_binary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int);


template<typename dtype>
void dilation_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    morph_binary_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, DILATION, flag_verbose);
}
template void dilation_binary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void dilation_binary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, 
                                        const int, const int);
template void dilation_binary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, 
                                        const int, const int);


template<typename dtype>
void closing_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose){
   
   MorphChain closing = {DILATION, EROSION};

   morph_chain_binary_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                             closing, flag_verbose);
}
template void closing_binary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void closing_binary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void closing_binary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int);


template<typename dtype>
void opening_binary(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose){
   
   MorphChain opening = {EROSION, DILATION};

   morph_chain_binary_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                             opening, flag_verbose);
}
template void opening_binary<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void opening_binary<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void opening_binary<u_int16_t>(u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int);

 