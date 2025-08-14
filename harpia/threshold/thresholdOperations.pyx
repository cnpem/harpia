cimport cython
from cython.parallel import prange
cimport numpy as np
import numpy as np
from libcpp cimport bool
from harpia.common import Size

#Define the fused type for numeric types : float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int

#Extern declaration for the Local Gaussian Threshold function from C / C++ library
cdef extern from "../../include/threshold/adaptative_gaussian.h":
    void local_gaussian_threshold[numeric](numeric* image, float* output, int rows, int cols, int depth, float sigma, float weight, bool type)

def local_gaussian(np.ndarray[numeric, ndim=3] image,
                   np.ndarray[np.float32_t, ndim=3] output = None,
                   float sigma = 1.0, float weight = 0.0, bool type = 1):
    """
    Apply a local Gaussian adaptive threshold to a 3D image.

    The threshold is computed as:  
    **T = Gaussian(image) - weight**

    Parameters:
        image (ndarray): Input 3D image of numeric type.
        output (ndarray, optional): Output array (float32) to store thresholded result.
                                    If None, a new array will be allocated.
        sigma (float): Standard deviation of the Gaussian kernel.
        weight (float): Bias subtracted from the filtered value.
        type (bool): If True, apply 3D filtering; if False, slice-by-slice.

    Returns:
        ndarray: The thresholded output image.
    """
    isize = Size(image)

    if output is None:
        output = np.empty((isize.z, isize.y, isize.x), dtype=np.float32)

    local_gaussian_threshold(&image[0, 0, 0],
                             &output[0, 0, 0],
                             isize.y, isize.x, isize.z,
                             sigma, weight, type)

    return output

#Local Mean Threshold
cdef extern from "../../include/threshold/adaptative_mean.h":
    void local_mean_threshold[numeric](numeric* image, float* output, float weight,
                                       int rows, int cols, int depth,
                                       int rows_kernel, int cols_kernel, int depth_kernel)

def local_mean(np.ndarray[numeric, ndim=3] image,
               np.ndarray[np.float32_t, ndim=3] output = None,
               float weight = 0.0,
               int windowSize = 3):
    """
    Apply a local mean threshold to a 3D image.

    The threshold is computed as:
    **T = LocalMean(image neighborhood) - weight**

    Parameters:
        image (ndarray): Input 3D image of numeric type.
        output (ndarray, optional): Output binary image (float32).
                                    If None, a new array will be allocated.
        weight (float): Constant bias to subtract from the local mean.
        windowSize (int): Size of the local neighborhood in all three dimensions (isotropic).

    Returns:
        ndarray: Thresholded binary result.
    """
    isize = Size(image)

    if output is None:
        output = np.empty((isize.z, isize.y, isize.x), dtype=np.float32)

    local_mean_threshold(&image[0, 0, 0],
                         &output[0, 0, 0],
                         weight,
                         isize.y, isize.x, isize.z,
                         windowSize, windowSize, windowSize)

    return output

#Niblack Threshold
cdef extern from "../../include/threshold/niblack.h":
    void niblack_threshold[numeric](numeric* image, float* output, float weight,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)

def niblack(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output = None,
            float weight = 0.0,
            int windowSize = 3):
    """
    Apply a Niblack threshold to a 3D image.

    The threshold is computed as:
    **T = local_mean + weight × local_stddev**

    Parameters:
        image (ndarray): Input 3D image of numeric type.
        output (ndarray, optional): Output array (float32) to store the result.
                                    If None, it will be auto-allocated.
        weight (float): Scaling factor for the local standard deviation.
        windowSize (int): Size of the cubic neighborhood used for computing local stats.

    Returns:
        ndarray: Binary output image after Niblack thresholding.
    """
    isize = Size(image)

    if output is None:
        output = np.empty((isize.z, isize.y, isize.x), dtype=np.float32)

    niblack_threshold(&image[0, 0, 0],
                      &output[0, 0, 0],
                      weight,
                      isize.y, isize.x, isize.z,
                      windowSize, windowSize, windowSize)

    return output

#Sauvola Threshold
cdef extern from "../../include/threshold/sauvola.h":
    void sauvola_threshold[numeric](numeric* image, float* output, float weight, float range,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)

def sauvola(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output = None,
            float weight = 0.0,
            float range = 1.0,
            int windowSize = 3):
    """
    Apply a Sauvola threshold to a 3D image.

    The threshold is computed as:
    **T = local_mean × (1 + weight × (local_stddev / range - 1))**

    Parameters:
        image (ndarray): Input 3D image of numeric type.
        output (ndarray, optional): Output array (float32) to store the thresholded result.
                                    If None, a new array is allocated automatically.
        weight (float): Multiplier for the local standard deviation.
        range (float): Expected dynamic range of the standard deviation (typically 128 or 1).
        windowSize (int): Size of the local cubic window used for thresholding.

    Returns:
        ndarray: Binary output image after Sauvola thresholding.
    """
    isize = Size(image)

    if output is None:
        output = np.empty((isize.z, isize.y, isize.x), dtype=np.float32)

    sauvola_threshold(&image[0, 0, 0],
                      &output[0, 0, 0],
                      weight, range,
                      isize.y, isize.x, isize.z,
                      windowSize, windowSize, windowSize)

    return output


# Otsu Threshold
cdef extern from "../../include/threshold/otsu.h":
    int otsu_threshold_value(int *histogramCounts, int nbins)

def otsu(np.ndarray[int, ndim=1] histogramCounts, int nbins):
    """
    Apply the Otsu threshold to a 1D histogram and compute the optimal threshold.

    Parameters:
        histogram (np.ndarray[int32_t, ndim=1]): Input 1D histogram array.
        bins (int): Number of bins in the histogram.
        a (float): Minimum value of the range.
        b (float): Maximum value of the range.
        totalPixels (int): Total number of pixels in the image.

    Returns:
        int: The optimal threshold value.
    """

    # Call the Otsu thresholding function
    return otsu_threshold_value(&histogramCounts[0], nbins)