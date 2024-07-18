cimport cython
cimport numpy as np
from libcpp cimport bool

# Define the fused type
ctypedef fused numeric:
    float
    int
    unsigned int
# Declare the external C functions using the fused type

# Gaussian filter
cdef extern from "../../include/filters/gaussian_filter.h":
    void gaussian_filtering[numeric] (numeric* image, float* output, int rows, int cols, int depth, float sigma, bool type)

def gaussian(np.ndarray[numeric, ndim=3] image, np.ndarray[np.float32_t, ndim=3] output,
             int rows, int cols, int depth,
             float sigma, bool type):
    return gaussian_filtering(&image[0,0,0], &output[0,0,0],
                              rows, cols, depth,
                              sigma, type)

# Mean filter
cdef extern from "../../include/filters/mean_filter.h":
    void mean_filtering[numeric] (numeric* image, float* output,
                                  int rows, int cols, int depth,
                                  int rows_kernel, int cols_kernel, int depth_kernel)

def mean(np.ndarray[numeric, ndim=3] image,
         np.ndarray[np.float32_t, ndim=3] output,
         int rows, int cols, int depth,
         int rows_kernel, int cols_kernel, int depth_kernel):
    return mean_filtering(&image[0,0,0],
                          &output[0,0,0],
                          rows, cols, depth,
                          rows_kernel, cols_kernel, depth_kernel)

# LoG filter
cdef extern from "../../include/filters/log_filter.h":
    void log_filtering[numeric] (numeric* image, float* output,
                                 int rows, int cols, int depth,
                                 bool type)

def LoG(np.ndarray[numeric, ndim=3] image,
        np.ndarray[np.float32_t, ndim=3] output,
        int rows, int cols, int depth,
        bool type):
    return log_filtering(&image[0,0,0],
                         &output[0,0,0],
                         rows, cols, depth,
                         type)

# Unsharp mask filter
cdef extern from "../../include/filters/unsharp_mask_filter.h":
    void unsharp_mask_filtering[numeric] (numeric* image, float* output,
                                          int rows, int cols, int depth,
                                          float sigma, float amount, float threshold, bool type)

def unsharp_mask(np.ndarray[numeric, ndim=3] image,
                 np.ndarray[np.float32_t, ndim=3] output,
                 int rows, int cols, int depth,
                 float sigma, float amount, float threshold, bool type):
    return unsharp_mask_filtering(&image[0,0,0],
                                  &output[0,0,0],
                                  rows, cols, depth,
                                  sigma, amount, threshold, type)

# Sobel filter
cdef extern from "../../include/filters/sobel_filter.h":
    void sobel_filtering[numeric] (numeric* image, float* output,
                                   int rows, int cols, int depth, bool type)

def sobel(np.ndarray[numeric, ndim=3] image,
          np.ndarray[np.float32_t, ndim=3] output,
          int rows, int cols, int depth, bool type):
    return sobel_filtering(&image[0,0,0],
                           &output[0,0,0],
                           rows, cols, depth, type)

# Prewitt filter
cdef extern from "../../include/filters/prewitt_filter.h":
    void prewitt_filtering[numeric] (numeric* image, float* output,
                                     int rows, int cols, int depth, bool type)

def prewitt(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output,
            int rows, int cols, int depth, bool type):
    return prewitt_filtering(&image[0,0,0],
                             &output[0,0,0],
                             rows, cols, depth, type)


# Canny filter
cdef extern from "../../include/filters/canny_filter.h":
    void canny_filtering[numeric] (numeric* image, float* output,
                                   int rows, int cols, int depth,
                                   float sigma, float low_threshold, float high_threshold)

def canny(np.ndarray[numeric, ndim=3] image,
          np.ndarray[np.float32_t, ndim=3] output,
          int rows, int cols, int depth,
          float sigma, float low_threshold, float high_threshold):
    return canny_filtering(&image[0,0,0],
                           &output[0,0,0],
                           rows, cols, depth,
                           sigma, low_threshold, high_threshold)
