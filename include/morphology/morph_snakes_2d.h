#ifndef MORPH_SNAKES_2D_H
#define MORPH_SNAKES_2D_H

/**
 * Performs the Morphological Geodesic Active Contour (MorphGAC) algorithm.
 * 
 * This function uses morphological operators for geodesic active contour segmentation. 
 * The algorithm is designed for segmenting objects with visible but potentially noisy or broken boundaries.
 *
 * @param hostImage A pointer to the preprocessed image array, with dimensions (ysize, xsize), 
 *        where pixel values are stored as floats.
 * @param initLs A pointer to the initial level set, represented as a boolean array of the same 
 *        dimensions as hostImage.
 * @param iterations Number of iterations to perform.
 * @param balloonForce A float representing the balloon force, which controls the contour expansion or shrinkage.
 *        Positive values expand the contour, negative values shrink it, and zero disables the force.
 * @param threshold A float threshold for stopping contour evolution in low-value areas of the image.
 * @param smoothing Number of smoothing iterations to apply in each evolution step.
 * @param hostOutput A pointer to the output boolean array, storing the final level set (segmentation).
 * @param xsize Width of the image in pixels.
 * @param ysize Height of the image in pixels.
 * @param flag_verbose Verbosity flag. If set to a non-zero value, the function will print
 *        the grid and block dimensions used for kernel execution to the console. This
 *        is useful for debugging and performance analysis to understand how the computation
 *        is distributed across CUDA threads.
 */
void morph_geodesic_active_contour(float* hostImage, bool* initLs, const int iterations, const float balloonForce, 
                                   const float threshold, const int smoothing, bool* hostOutput,
                                   const int xsize, const int ysize, const int flag_verbose);

/**
 * Performs the Morphological Active Contours without Edges (MorphACWE) algorithm.
 *
 * This function uses morphological operators for active contour segmentation without edge information,
 * based on region differences inside and outside the contour.
 *
 * @param hostImage A pointer to the grayscale image array, with dimensions (ysize, xsize).
 * @param initLs A pointer to the initial level set, represented as a boolean array with dimensions matching hostImage.
 * @param iterations Number of iterations to perform.
 * @param lambda1 Weight for the outer region. Higher values favor a broader range of values in the outer region.
 * @param lambda2 Weight for the inner region. Higher values favor a broader range of values in the inner region.
 * @param smoothing Number of smoothing iterations to apply in each evolution step.
 * @param hostOutput A pointer to the output boolean array storing the final level set (segmentation).
 * @param xsize Width of the image in pixels.
 * @param ysize Height of the image in pixels.
 * @param flag_verbose Verbosity flag. If set to a non-zero value, the function will print
 *        the grid and block dimensions used for kernel execution to the console. This
 *        is useful for debugging and performance analysis to understand how the computation
 *        is distributed across CUDA threads.
 */
void morph_chan_vese(float* hostImage, bool* initLs, const int iterations, const float lambda1, const float lambda2, 
                     const int smoothing, bool* hostOutput, const int xsize, const int ysize, const int flag_verbose);

#endif  // MORPH_SNAKES_2D_H
