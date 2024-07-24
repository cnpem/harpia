cimport cython
cimport numpy as np
from libcpp cimport bool

# Define the fused type
ctypedef fused numeric:
    float
    int
    unsigned int

# Local Gaussian Threshold
cdef extern from "../../include/threshold/adaptative_gaussian.h":
    void local_gaussian_threshold[numeric](numeric* image, float* output, int rows, int cols, int depth, float sigma, float weight, bool type)

def local_gaussian(np.ndarray[numeric, ndim=3] image,
                   np.ndarray[np.float32_t, ndim=3] output,
                   int rows, int cols, int depth,
                   float sigma, float weight, bool type):
    return local_gaussian_threshold(&image[0,0,0],
                                    &output[0,0,0],
                                    rows, cols, depth,
                                    sigma, weight, type)

# Local Mean Threshold
cdef extern from "../../include/threshold/adaptative_mean.h":
    void local_mean_threshold[numeric](numeric* image, float* output, float weight,
                                       int rows, int cols, int depth,
                                       int rows_kernel, int cols_kernel, int depth_kernel)

def local_mean(np.ndarray[numeric, ndim=3] image,
               np.ndarray[np.float32_t, ndim=3] output,
               float weight,
               int rows, int cols, int depth,
               int rows_kernel, int cols_kernel, int depth_kernel):
    return local_mean_threshold(&image[0,0,0],
                                &output[0,0,0],
                                weight,
                                rows, cols, depth,
                                rows_kernel, cols_kernel, depth_kernel)

# Niblack Threshold
cdef extern from "../../include/threshold/niblack.h":
    void niblack_threshold[numeric](numeric* image, float* output, float weight,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)

def niblack(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output,
            float weight,
            int rows, int cols, int depth,
            int rows_kernel, int cols_kernel, int depth_kernel):
    return niblack_threshold(&image[0,0,0],
                             &output[0,0,0],
                             weight,
                             rows, cols, depth,
                             rows_kernel, cols_kernel, depth_kernel)

# Sauvola Threshold
cdef extern from "../../include/threshold/sauvola.h":
    void sauvola_threshold[numeric](numeric* image, float* output, float weight, float range,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)

def sauvola(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output,
            float weight, float range,
            int rows, int cols, int depth,
            int rows_kernel, int cols_kernel, int depth_kernel):
    return sauvola_threshold(&image[0,0,0],
                             &output[0,0,0],
                             weight, range,
                             rows, cols, depth,
                             rows_kernel, cols_kernel, depth_kernel)
