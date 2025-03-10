#ifndef MORPH_binary_on_device_OPS_H
#define MORPH_binary_on_device_OPS_H

#include "morphology.h"

/**
 * @brief Performs binary erosion on the entire image using the GPU or CPU.
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
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void erosion_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

/**
 * @brief Performs binary dilation on the entire image using the GPU or CPU.
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
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void dilation_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                     const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                     int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

/**
 * @brief Performs binary closing on the entire image using the GPU or CPU.
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
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void closing_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

/**
 * @brief Performs binary openig on the entire image using the GPU or CPU.
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
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void opening_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                    const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                    int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

/**
 * @brief Performs binary smooth on the entire image using the GPU or CPU.
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
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void smooth_binary(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                   const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                   int kernel_ysize, int kernel_zsize, float gpuMemory, bool gpu);

/**
 * @brief Perform geodesic erosion operation on the entire image using the GPU or CPU.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostMask Pointer to the mask image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void geodesic_erosion_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                             const int ysize, const int zsize, const int flag_verbose,
                             float gpuMemory, bool gpu);

/**
 * @brief Perform geodesic dilation operation on the entire image using the GPU or CPU. 
 *
 * @tparam dtype The data type of the image.
 * @param hostImage Pointer to the input image on the host (marker image).
 * @param hostMask Pointer to the mask image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void geodesic_dilation_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              float gpuMemory, bool gpu);

/**
 * @brief Perform morphological reconstruction operation using the GPU or CPU.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostMask Pointer to the mask image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param operation Morphological operation to be performed (erosion or dilation).
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void reconstruction_binary(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                           const int ysize, const int zsize, const int flag_verbose,
                           MorphOp operation, bool gpu);

/**
 * @brief Fill holes in the binary image using the GPU or CPU.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param padding Padding size for hole filling.
 * @param flag_verbose Flag for verbose output.
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param gpu Flag to indicate whether to use GPU (true) or CPU (false).
 */
template <typename dtype>
void fill_holes(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, int padding, const int flag_verbose, float gpuMemory, bool gpu);

#endif  // MORPH_binary_on_device_OPS_H