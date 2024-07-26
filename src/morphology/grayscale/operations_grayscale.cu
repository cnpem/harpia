#include "../../../include/morphology/morph_grayscale.h"
#include "../../../include/morphology/morph_chain_grayscale.h"
#include "../../../include/morphology/operations_grayscale.h"

template<typename dtype>
void erosion_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    morph_grayscale_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, EROSION, flag_verbose);
}
template void erosion_grayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void erosion_grayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void erosion_grayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);


template<typename dtype>
void dilation_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    morph_grayscale_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                        xsize, ysize, zsize, DILATION, flag_verbose);
}
template void dilation_grayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void dilation_grayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void dilation_grayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);


template<typename dtype>
void closing_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose){
   
   MorphChain closing = {DILATION, EROSION};

   morph_chain_grayscale_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                              closing, flag_verbose);
}
template void closing_grayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void closing_grayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void closing_grayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);

template<typename dtype>
void opening_grayscale(dtype *hostImage, dtype *hostOutput, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                        const int xsize, const int ysize, const int zsize, const int flag_verbose){
   
   MorphChain opening = {EROSION, DILATION};

   morph_chain_grayscale_on_device(hostImage, hostOutput, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize,
                               opening, flag_verbose);
}
template void opening_grayscale<int>(int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void opening_grayscale<u_int32_t>(u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void opening_grayscale<float>(float *, float *, int *, int, int, int, const int, const int, const int, const int);

 