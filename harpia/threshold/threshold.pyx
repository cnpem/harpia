cimport cython
cimport numpy as np
from libcpp cimport bool

# Define the fused type for numeric types: float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int

# Extern declaration for the Local Gaussian Threshold function from C/C++ library
cdef extern from "../../include/threshold/adaptative_gaussian.h":
    void local_gaussian_threshold[numeric](numeric* image, float* output, int rows, int cols, int depth, float sigma, float weight, bool type)

def local_gaussian(np.ndarray[numeric, ndim=3] image,
                   np.ndarray[np.float32_t, ndim=3] output,
                   int rows, int cols, int depth,
                   float sigma, float weight, bool type):
    """
    Apply a local Gaussian threshold to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the thresholded result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        sigma (float): Standard deviation for Gaussian kernel.
        weight (float): Weight parameter for thresholding.
        type (bool): Type of thresholding (specific to implementation).

    Returns:
        None
    """
    return local_gaussian_threshold(&image[0,0,0],
                                    &output[0,0,0],
                                    rows, cols, depth,
                                    sigma, weight, type)

# Extern declaration for the Local Mean Threshold function from C/C++ library
cdef extern from "../../include/threshold/adaptative_mean.h":
    void local_mean_threshold[numeric](numeric* image, float* output, float weight,
                                       int rows, int cols, int depth,
                                       int rows_kernel, int cols_kernel, int depth_kernel)

def local_mean(np.ndarray[numeric, ndim=3] image,
               np.ndarray[np.float32_t, ndim=3] output,
               float weight,
               int rows, int cols, int depth,
               int rows_kernel, int cols_kernel, int depth_kernel):
    """
    Apply a local mean threshold to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the thresholded result.
        weight (float): Weight parameter for thresholding.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        rows_kernel (int): Number of rows in the kernel.
        cols_kernel (int): Number of columns in the kernel.
        depth_kernel (int): Number of depth slices in the kernel.

    Returns:
        None
    """
    return local_mean_threshold(&image[0,0,0],
                                &output[0,0,0],
                                weight,
                                rows, cols, depth,
                                rows_kernel, cols_kernel, depth_kernel)

# Extern declaration for the Niblack Threshold function from C/C++ library
cdef extern from "../../include/threshold/niblack.h":
    void niblack_threshold[numeric](numeric* image, float* output, float weight,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)

def niblack(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output,
            float weight,
            int rows, int cols, int depth,
            int rows_kernel, int cols_kernel, int depth_kernel):
    """
    Apply a Niblack threshold to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the thresholded result.
        weight (float): Weight parameter for thresholding.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        rows_kernel (int): Number of rows in the kernel.
        cols_kernel (int): Number of columns in the kernel.
        depth_kernel (int): Number of depth slices in the kernel.

    Returns:
        None
    """
    return niblack_threshold(&image[0,0,0],
                             &output[0,0,0],
                             weight,
                             rows, cols, depth,
                             rows_kernel, cols_kernel, depth_kernel)

# Extern declaration for the Sauvola Threshold function from C/C++ library
cdef extern from "../../include/threshold/sauvola.h":
    void sauvola_threshold[numeric](numeric* image, float* output, float weight, float range,
                                    int rows, int cols, int depth,
                                    int rows_kernel, int cols_kernel, int depth_kernel)

def sauvola(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output,
            float weight, float range,
            int rows, int cols, int depth,
            int rows_kernel, int cols_kernel, int depth_kernel):
    """
    Apply a Sauvola threshold to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the thresholded result.
        weight (float): Weight parameter for thresholding.
        range (float): Range parameter for thresholding.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        rows_kernel (int): Number of rows in the kernel.
        cols_kernel (int): Number of columns in the kernel.
        depth_kernel (int): Number of depth slices in the kernel.

    Returns:
        None
    """
    return sauvola_threshold(&image[0,0,0],
                             &output[0,0,0],
                             weight, range,
                             rows, cols, depth,
                             rows_kernel, cols_kernel, depth_kernel)
