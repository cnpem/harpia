#ifndef OTSU_THRESHOLD_H
#define OTSU_THRESHOLD_H

#include <cuda_runtime.h>

// Function to calculate the optimal Otsu threshold value
int otsu_threshold_value(int *histogramCounts, int nbins);

#endif // OTSU_THRESHOLD_H
