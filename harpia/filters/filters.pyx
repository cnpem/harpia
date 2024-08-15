cimport cython
cimport numpy as np
from libcpp cimport bool

# Define the fused type for numeric types: float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int

# Extern declaration for the Gaussian filtering function from C/C++ library
cdef extern from "../../include/filters/gaussian_filter.h":
    void gaussian_filtering[numeric] (numeric* hostImage, float* hostOutput, int xsize, int ysize, int zsize, float sigma, bool type)

def gaussian(np.ndarray[numeric, ndim=3] hostImage, np.ndarray[np.float32_t, ndim=3] hostOutput,
             int xsize, int ysize, int zsize,
             float sigma, bool type):
    """
    Apply a Gaussian filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        sigma (float): Standard deviation for Gaussian kernel.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return gaussian_filtering(&hostImage[0,0,0], &hostOutput[0,0,0],
                              xsize, ysize, zsize,
                              sigma, type)

# Extern declaration for the Mean filtering function from C/C++ library
cdef extern from "../../include/filters/mean_filter.h":
    void mean_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                  int xsize, int ysize, int zsize,
                                  int nx, int ny, int nz)

def mean(np.ndarray[numeric, ndim=3] hostImage,
         np.ndarray[np.float32_t, ndim=3] hostOutput,
         int xsize, int ysize, int zsize,
         int nx, int ny, int nz):
    """
    Apply a Mean filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        nx (int): Number of rows in the kernel.
        ny (int): Number of columns in the kernel.
        nz (int): Number of slices in the kernel.

    Returns:
        None
    """
    return mean_filtering(&hostImage[0,0,0],
                          &hostOutput[0,0,0],
                          xsize, ysize, zsize,
                          nx, ny, nz)

# Extern declaration for the LoG filtering function from C/C++ library
cdef extern from "../../include/filters/log_filter.h":
    void log_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                 int xsize, int ysize, int zsize,
                                 bool type)

def LoG(np.ndarray[numeric, ndim=3] hostImage,
        np.ndarray[np.float32_t, ndim=3] hostOutput,
        int xsize, int ysize, int zsize,
        bool type):
    """
    Apply a Laplacian of Gaussian (LoG) filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return log_filtering(&hostImage[0,0,0],
                         &hostOutput[0,0,0],
                         xsize, ysize, zsize,
                         type)

# Extern declaration for the Unsharp Mask filtering function from C/C++ library
cdef extern from "../../include/filters/unsharp_mask_filter.h":
    void unsharp_mask_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                          int xsize, int ysize, int zsize,
                                          float sigma, float amount, float threshold, bool type)

def unsharp_mask(np.ndarray[numeric, ndim=3] hostImage,
                 np.ndarray[np.float32_t, ndim=3] hostOutput,
                 int xsize, int ysize, int zsize,
                 float sigma, float amount, float threshold, bool type):
    """
    Apply an Unsharp Mask filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        sigma (float): Standard deviation for Gaussian kernel.
        amount (float): Amount of the unsharp mask.
        threshold (float): Threshold for the unsharp mask.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return unsharp_mask_filtering(&hostImage[0,0,0],
                                  &hostOutput[0,0,0],
                                  xsize, ysize, zsize,
                                  sigma, amount, threshold, type)

# Extern declaration for the Sobel filtering function from C/C++ library
cdef extern from "../../include/filters/sobel_filter.h":
    void sobel_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                   int xsize, int ysize, int zsize, bool type)

def sobel(np.ndarray[numeric, ndim=3] hostImage,
          np.ndarray[np.float32_t, ndim=3] hostOutput,
          int xsize, int ysize, int zsize, bool type):
    """
    Apply a Sobel filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return sobel_filtering(&hostImage[0,0,0],
                           &hostOutput[0,0,0],
                           xsize, ysize, zsize, type)

# Extern declaration for the Prewitt filtering function from C/C++ library
cdef extern from "../../include/filters/prewitt_filter.h":
    void prewitt_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                     int xsize, int ysize, int zsize, bool type)

def prewitt(np.ndarray[numeric, ndim=3] hostImage,
            np.ndarray[np.float32_t, ndim=3] hostOutput,
            int xsize, int ysize, int zsize, bool type):
    """
    Apply a Prewitt filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return prewitt_filtering(&hostImage[0,0,0],
                             &hostOutput[0,0,0],
                             xsize, ysize, zsize, type)

# Extern declaration for the Canny filtering function from C/C++ library
cdef extern from "../../include/filters/canny_filter.h":
    void canny_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                   int xsize, int ysize, int zsize,
                                   float sigma, float low_threshold, float high_threshold)

def canny(np.ndarray[numeric, ndim=3] hostImage,
          np.ndarray[np.float32_t, ndim=3] hostOutput,
          int xsize, int ysize, int zsize,
          float sigma, float low_threshold, float high_threshold):
    """
    Apply a Canny filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        sigma (float): Standard deviation for Gaussian kernel.
        low_threshold (float): Lower threshold for the hysteresis procedure.
        high_threshold (float): Upper threshold for the hysteresis procedure.

    Returns:
        None
    """
    return canny_filtering(&hostImage[0,0,0],
                           &hostOutput[0,0,0],
                           xsize, ysize, zsize,
                           sigma, low_threshold, high_threshold)


# Extern declaration for the median filtering function from C/C++ library
cdef extern from "../../include/filters/median_filter.h":
    void median_filtering[numeric] (numeric* hostImage, numeric* hostOutput,
                                  int xsize, int ysize, int zsize,
                                  int nx, int ny, int nz)

def median(np.ndarray[numeric, ndim=3] hostImage,
         np.ndarray[numeric, ndim=3] hostOutput,
         int xsize, int ysize, int zsize,
         int nx, int ny, int nz):
    """
    Apply a median filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        zsize (int): Number of slices in the hostImage.
        nx (int): Number of rows in the kernel.
        ny (int): Number of columns in the kernel.
        nz (int): Number of slices in the kernel.

    Returns:
        None
    """
    return median_filtering(&hostImage[0,0,0],
                          &hostOutput[0,0,0],
                          xsize, ysize, zsize,
                          nx, ny, nz)
