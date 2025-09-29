#ifndef MORPH_grayscale_on_device_OPS_H
#define MORPH_grayscale_on_device_OPS_H

#include "morphology.h"

/**
 * @brief Performs grayscale erosion on the entire image using the GPU or CPU.
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
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorKernel(), morph_grayscale_on_device(), morph_grayscale_on_host()
 */
template <typename dtype>
void erosion_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus);

/**
 * @brief Performs grayscale dilation on the entire image using the GPU or CPU.
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
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorKernel(), morph_grayscale_on_device(), morph_grayscale_on_host()
 */
template <typename dtype>
void dilation_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                        const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                        int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus);

/**
 * @brief Performs grayscale closing on the entire image using the GPU or CPU.
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
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorKernel(), morph_chain_grayscale_on_device(), morph_chain_grayscale_on_host()
 */
template <typename dtype>
void closing_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus);
/**
 * @brief Performs grayscale openig on the entire image using the GPU or CPU.
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
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorKernel(), morph_chain_grayscale_on_device(), morph_chain_grayscale_on_host()
 */
template <typename dtype>
void opening_grayscale(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                       const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                       int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus);

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
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorGeodesic(), geodesic_morph_grayscale_on_device(), geodesic_morph_grayscale_on_host()
 */
template <typename dtype>
void geodesic_erosion_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                const int xsize, const int ysize, const int zsize,
                                const int flag_verbose, float gpuMemory, int ngpus);
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
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorGeodesic(), geodesic_morph_grayscale_on_device(), geodesic_morph_grayscale_on_host()
 */
template <typename dtype>
void geodesic_dilation_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput,
                                 const int xsize, const int ysize, const int zsize,
                                 const int flag_verbose, float gpuMemory, int ngpus);

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
 * @param ngpus Whether to execute on GPU or CPU. 
 *              If ngpus = 0, CPU execution is selected. 
 *              Otherwise, the function executes on GPU.
 *
 * @see reconstruction_grayscale_on_device(), reconstruction_grayscale_on_host()
 */
template <typename dtype>
void reconstruction_grayscale(dtype* hostImage, dtype* hostMask, dtype* hostOutput, const int xsize,
                              const int ysize, const int zsize, const int flag_verbose,
                              MorphOp operation, int ngpus);

/**
 * @brief Performs bottom-hat transformation on the entire image using the GPU or CPU.
 *
 * The bottom-hat transformation is defined as the difference between the closing 
 * of the image and the original image. It is useful for highlighting dark features 
 * on a bright background.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorKernel(), bottom_hat_on_device(), bottom_hat_on_host()
 */
template <typename dtype>
void bottom_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                int kernel_ysize, int kernel_zsize, float gpuMemory, int ngpus);

/**
 * @brief Performs top-hat transformation on the entire image using the GPU or CPU.
 *
 * The top-hat transformation is defined as the difference between the original 
 * image and its opening. It is useful for enhancing bright features on a dark background.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param gpuMemory Percentage of free GPU memory to be used, with values ranging from 0 to 1.
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.
 *              If ngpus = 0, CPU execution is selected. 
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 *
 * @see chunkedExecutorKernel(), top_hat_on_device(), top_hat_on_host()
 */
template <typename dtype>
void top_hat(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize, const int zsize,
             const int flag_verbose, int* kernel, int kernel_xsize, int kernel_ysize,
             int kernel_zsize, float gpuMemory, int ngpus);

/**
 * @brief Performs top-hat transformation using morphological reconstruction.
 *
 * Instead of a standard morphological opening, this version of top-hat applies 
 * morphological reconstruction to enhance features more effectively while preserving 
 * the original structure of objects.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param ngpus Whether to execute on GPU or CPU. 
 *              If ngpus = 0, CPU execution is selected. 
 *              Otherwise, the function executes on GPU.
 *
 * @see top_hat_reconstruction_on_device(), top_hat_reconstruction_on_host
 */
template <typename dtype>
void top_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                            const int zsize, const int flag_verbose, int* kernel, int kernel_xsize,
                            int kernel_ysize, int kernel_zsize, int ngpus);

/**
 * @brief Performs bottom-hat transformation using morphological reconstruction.
 *
 * This method applies morphological reconstruction instead of a standard closing 
 * operation to highlight dark regions more effectively while preserving structural details.
 *
 * @tparam dtype Data type of the image.
 * @param hostImage Pointer to the input image on the host.
 * @param hostOutput Pointer to the output image on the host.
 * @param xsize Size of the image in the x-dimension.
 * @param ysize Size of the image in the y-dimension.
 * @param zsize Size of the image in the z-dimension.
 * @param flag_verbose Flag for verbose output.
 * @param kernel Pointer to the morphological kernel.
 * @param kernel_xsize Size of the kernel in the x-dimension.
 * @param kernel_ysize Size of the kernel in the y-dimension.
 * @param kernel_zsize Size of the kernel in the z-dimension.
 * @param ngpus Whether to execute on GPU or CPU. 
 *              If ngpus = 0, CPU execution is selected. 
 *              Otherwise, the function executes on GPU.
 *
 * @see bottom_hat_reconstruction_on_device(), bottom_hat_reconstruction_on_host
 */
template <typename dtype>
void bottom_hat_reconstruction(dtype* hostImage, dtype* hostOutput, const int xsize,
                               const int ysize, const int zsize, const int flag_verbose,
                               int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize,
                               int ngpus);

#endif  // MORPH_grayscale_on_device_OPS_H