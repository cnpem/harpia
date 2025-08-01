#ifndef CHUNKED_EXECUTOR_H
#define CHUNKED_EXECUTOR_H

#include <stdio.h>
#include <iostream>

/**
 * @brief Wrapper function to execute a CUDA function in chunks.
 * 
 * Splits a 3D image into smaller chunks based on available GPU memory and processes each chunk separately.
 * The image is divided along the z-axis to simplify memory management and computation.
 * Each chunk is passed to the specified CUDA function for processing, ensuring efficient execution 
 * while preventing memory overflows.
 *
 * @tparam Func Type of the CUDA function to execute.
 * @tparam dtype Data type of the image (e.g., uint8_t, float).
 * @tparam Args Variadic template arguments for additional parameters to the CUDA function.
 * @param func The CUDA function to execute.
 * @param ncopies Number of copies of the image data needed to execute the algorithm.
 * @param safetyMargin Fraction of available GPU memory to use (0-1).
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.  
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 * @param image Pointer to the input image data.
 * @param output Pointer to the output image data.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image.
 * @param verbose Verbosity flag (1 for debug output, 0 for silent execution).
 * @param args Additional arguments passed to the CUDA function.
 */
template <typename Func, typename in_dtype, typename out_dtype, typename... Args>
void chunkedExecutor(Func func, int ncopies, const float safetyMargin, int ngpus, 
                     in_dtype* image, out_dtype* output, const int xsize, const int ysize, const int zsize, 
                     const int verbose, Args... args);

/**
 * @brief Specialized chunked execution for morphological operations using a kernel.
 *
 * This function is optimized for executing morphological operations that use a kernel.
 * It splits the 3D image into smaller chunks based on available GPU memory and processes
 * each chunk separately using the specified CUDA function.
 *
 * When a kernel operation is applied, padding is required to ensure correct border handling.
 * In cases where two consecutive kernel operations are performed, the first execution applies
 * double padding to guarantee that the second operation also has the necessary padding.
 *
 * @tparam Func Type of the CUDA function to execute.
 * @tparam dtype Data type of the image.
 * @tparam Args Variadic template arguments for additional parameters.
 * @param func The CUDA function to execute.
 * @param ncopies Number of copies of the image data needed to execute the algorithm.
 * @param safetyMargin Fraction of available GPU memory to use.
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.  
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 * @param kernelOperations Number of kernel operations to be performed.
 * @param image Pointer to the input image data.
 * @param output Pointer to the output image data.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image.
 * @param verbose Verbosity flag.
 * @param kernel Pointer to the kernel data.
 * @param kernel_xsize Width of the kernel.
 * @param kernel_ysize Height of the kernel.
 * @param kernel_zsize Depth of the kernel.
 * @param args Additional arguments passed to the CUDA function.
 */
template <typename Func, typename in_dtype, typename out_dtype, typename kernel_dtype,  typename... Args>
void chunkedExecutorKernel(Func func, int ncopies, const float safetyMargin, int ngpus, 
                           const int kernelOperations, in_dtype* image, out_dtype* output, const int xsize,
                           const int ysize, const int zsize, const int verbose, kernel_dtype* kernel,
                           int kernel_xsize, int kernel_ysize, int kernel_zsize, Args... args);

/**
 * @brief Specialized chunked execution for executing geodesic morphological operations.
 *
 * This function is designed for geodesic operations that use an 8-connectivity fixed kernel of ones
 * and require both an image and a mask as input.
 *
 * @tparam Func Type of the CUDA function to execute.
 * @tparam dtype Data type of the image.
 * @tparam Args Variadic template arguments for additional parameters.
 * @param func The CUDA function to execute.
 * @param ncopies Number of copies of the image data.
 * @param safetyMargin Fraction of available GPU memory to use.
 * @param ngpus The number of GPUs to use. 
 *              If ngpus < 1, all available GPUs are used.  
 *              If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data.
 * @param output Pointer to the output image data.
 * @param xsize Width of the image.
 * @param ysize Height of the image.
 * @param zsize Depth of the image.
 * @param verbose Verbosity flag.
 * @param args Additional arguments passed to the CUDA function.
 */
template <typename Func, typename dtype, typename... Args>
void chunkedExecutorGeodesic(Func func, int ncopies, const float safetyMargin, int ngpus, 
                             dtype* image, dtype* mask, dtype* output, const int xsize, 
                             const int ysize, const int zsize, const int verbose, Args... args);

template <typename Func, typename dtype, typename... Args>
void chunkedExecutorFillHoles(Func func, int ncopies, const float safetyMargin, int ngpus, 
                              dtype* image, dtype* output, int padding, const int xsize, 
                              const int ysize, const int zsize, const int verbose, Args... args);

template <typename Func, typename in_dtype, typename out_dtype, typename... Args>
void chunkedExecutorPixelFeatures(Func func, int ncopies, int nFeatures, const float safetyMargin, int ngpus, 
                     in_dtype* image, out_dtype* output, const int xsize, const int ysize, const int zsize, 
                     const int verbose, Args... args);

template <typename Func, typename dtype, typename... Args>
void chunkedExecutorSuperpixelFeatures(Func func, int ncopies, const float safetyMargin, int ngpus, 
                             dtype* image, int* superpixel, float* output, 
                             const int xsize, const int ysize, const int zsize,
                             int nsuperpixels, int nfeatures, bool mean, bool min, bool max,
                             const int verbose, Args... args);


                             
// Includes the implementation to prevent linkage errors during compilation,  
// similar to defining the function in the header.
#include "../../src/chunkedExecutor/chunkedExecutor.cu"
#include "../../src/chunkedExecutor/chunkedExecutorKernel.cu"
#include "../../src/chunkedExecutor/chunkedExecutorGeodesic.cu"
#include "../../src/chunkedExecutor/chunkedExecutorFillHoles.cu"
#include "../../src/chunkedExecutor/chunkedExecutorPixel.cu"
#include "../../src/chunkedExecutor/chunkedExecutorSuperpixel.cu"

#endif  // CHUNKED_EXECUTOR_H


