#include "../../../include/morphology/geodesic_morph_binary.h"
#include "../../../include/morphology/morph_binary.h"
#include "../../../include/morphology/morph_chain_binary.h"
#include "../../../include/morphology/operations_binary.h"

/**
 * @brief Performs binary erosion on the input image.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
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

/**
 * @brief Performs binary dilation on the input image.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
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

/**
 * @brief Performs binary closing on the input image.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
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

/**
 * @brief Performs binary openig on the input image.
 * 
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 */
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

/**
 * @brief Perform geodesic erosion operation on the entire image using the GPU. This function is meant to be called from host
 * and slide the morph_binary kerel function through all pixels.
 * 
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param kernel Morphological operation kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 */
template<typename dtype>
void geodesic_erosion_binary(dtype *hostImage, dtype *hostOutput, dtype *hostMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    geodesic_morph_binary_on_device(hostImage, hostOutput, hostMask, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                xsize, ysize, zsize, EROSION, flag_verbose);
}
template void geodesic_erosion_binary<int>(int *, int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void geodesic_erosion_binary<u_int32_t>(u_int32_t *, u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void geodesic_erosion_binary<u_int16_t>(u_int16_t *, u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int);

/**
 * @brief Perform geodesic dilation operation on the entire image using the GPU. This function is meant to be called from host
 * and slide the morph_binary kerel function through all pixels.
 * 
 * @tparam dtype The data type of the image.
 * @param hostImage Input image on the host (corresponds to the marker image).
 * @param hostOutput Output image on the host.
 * @param hostMask Mask image on the host.
 * @param kernel Morphological operation kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Verbose flag to print grid and block dimensions.
 */
template<typename dtype>
void geodesic_dilation_binary(dtype *hostImage, dtype *hostOutput, dtype *hostMask, int *kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                 const int xsize, const int ysize, const int zsize, const int flag_verbose)
{
    geodesic_morph_binary_on_device(hostImage, hostOutput, hostMask, kernel, kernel_xsize, kernel_ysize, kernel_zsize, 
                xsize, ysize, zsize, DILATION, flag_verbose);
}
template void geodesic_dilation_binary<int>(int *, int *, int *, int *, int, int, int, const int, const int, const int, const int);
template void geodesic_dilation_binary<u_int32_t>(u_int32_t *, u_int32_t *, u_int32_t *, int *, int, int, int, const int, const int, const int, const int);
template void geodesic_dilation_binary<u_int16_t>(u_int16_t *, u_int16_t *, u_int16_t *, int *, int, int, int, const int, const int, const int, const int);
