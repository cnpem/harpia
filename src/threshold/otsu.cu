#include <iostream>
#include <vector>
#include <numeric>  // For std::accumulate

// Function to calculate the Otsu threshold
int otsu_threshold_value(int *histogram, int nbins) {
    int total_pixels = std::accumulate(histogram, histogram + nbins, 0);

    double sum1 = 0;
    for (int i = 0; i < nbins; i++) {
        sum1 += static_cast<double>(i) * histogram[i];
    }

    double sumB = 0;
    int wB = 0, threshold = 0;
    double max_variance = 0.0;

    for (int t = 0; t < nbins; t++) {
        wB += histogram[t];  // Weight of the background
        if (wB == 0) continue;

        int wF = total_pixels - wB;  // Weight of the foreground
        if (wF == 0) break;

        sumB += static_cast<double>(t) * histogram[t];  // Sum of background

        // Calculate foreground mean mF
        double mF = (sum1 - sumB) / wF;
        // Calculate background mean mB
        double mB = sumB / wB;

        // Calculate between-class variance
        double variance = static_cast<double>(wB) * wF * (mB - mF) * (mB - mF);

        // Update threshold if this variance is the maximum found
        if (variance >= max_variance) {
            max_variance = variance;
            threshold = t + 1;  // Align with MATLAB's one-based index
        }
    }

    return threshold;
}
