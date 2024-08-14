#ifndef ANISOTROPIC_DIFFUSION_H
#define ANISOTROPIC_DIFFUSION_H

#include <cmath>
#include <numeric>
#include <iostream>

/**
 * @brief Performs anisotropic diffusion on a 2D or 3D image.
 * 
 * This function applies the anisotropic diffusion algorithm to enhance images by reducing noise while preserving edges.
 * The function supports three different diffusion options that control the smoothing behavior.
 * 
 * @tparam dtype Data type of the image (e.g., uint8_t, float).
 * @param inputImage Pointer to the input 2D image data.
 * @param totalIterations Number of iterations to perform.
 * @param deltaT Time step size.
 * @param kappa Gradient modulus threshold that influences the conduction.
 * @param diffusionOption Choice of diffusion function:
 *                        1 - Exponential decay,
 *                        2 - Inverse quadratic decay,
 *                        3 - Hyperbolic tangent decay.
 * Option 3 is a faster implementation:
 * Mbarki, Zouhair, et al. "A new rapid auto-adapting diffusion function for adaptive anisotropic 
 * image de-noising and sharply conserved edges." Computers & Mathematics with Applications 74.8 (2017): 1751-1768.
 * @param numRows Number of rows in the image.
 * @param numCols Number of columns in the image.
 */
template<typename dtype>
void anisotropicDiffusion2D(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols);

template<typename dtype>
void anisotropicDiffusion3D(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols, int numSlices);

template<typename dtype>
void anisotropicDiffusion2DGPU(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols);

template<typename dtype>
void anisotropicDiffusion3DGPU(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols, int numSlices);



#endif
