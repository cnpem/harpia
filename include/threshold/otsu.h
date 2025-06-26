#ifndef OTSU_THRESHOLD_H
#define OTSU_THRESHOLD_H

#include <cuda_runtime.h>

/**
 * @brief Computes the optimal threshold using Otsu's method.
 *
 * This function calculates the threshold that maximizes the between-class variance
 * given a histogram of pixel intensities.
 *
 * @param[in] histogramCounts Pointer to the histogram array (of size `nbins`).
 * @param[in] nbins Number of bins in the histogram (e.g., 256 for 8-bit images).
 * @return Optimal threshold value (integer index in [0, nbins-1]).
 */
int otsu_threshold_value(int *histogramCounts, int nbins);

#endif // OTSU_THRESHOLD_H
