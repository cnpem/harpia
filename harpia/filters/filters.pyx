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
    void gaussian_filtering[numeric] (numeric* image, float* output, int rows, int cols, int depth, float sigma, bool type)

def gaussian(np.ndarray[numeric, ndim=3] image, np.ndarray[np.float32_t, ndim=3] output,
             int rows, int cols, int depth,
             float sigma, bool type):
    """
    Apply a Gaussian filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        sigma (float): Standard deviation for Gaussian kernel.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return gaussian_filtering(&image[0,0,0], &output[0,0,0],
                              rows, cols, depth,
                              sigma, type)

# Extern declaration for the Mean filtering function from C/C++ library
cdef extern from "../../include/filters/mean_filter.h":
    void mean_filtering[numeric] (numeric* image, float* output,
                                  int rows, int cols, int depth,
                                  int rows_kernel, int cols_kernel, int depth_kernel)

def mean(np.ndarray[numeric, ndim=3] image,
         np.ndarray[np.float32_t, ndim=3] output,
         int rows, int cols, int depth,
         int rows_kernel, int cols_kernel, int depth_kernel):
    """
    Apply a Mean filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        rows_kernel (int): Number of rows in the kernel.
        cols_kernel (int): Number of columns in the kernel.
        depth_kernel (int): Number of depth slices in the kernel.

    Returns:
        None
    """
    return mean_filtering(&image[0,0,0],
                          &output[0,0,0],
                          rows, cols, depth,
                          rows_kernel, cols_kernel, depth_kernel)

# Extern declaration for the LoG filtering function from C/C++ library
cdef extern from "../../include/filters/log_filter.h":
    void log_filtering[numeric] (numeric* image, float* output,
                                 int rows, int cols, int depth,
                                 bool type)

def LoG(np.ndarray[numeric, ndim=3] image,
        np.ndarray[np.float32_t, ndim=3] output,
        int rows, int cols, int depth,
        bool type):
    """
    Apply a Laplacian of Gaussian (LoG) filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return log_filtering(&image[0,0,0],
                         &output[0,0,0],
                         rows, cols, depth,
                         type)

# Extern declaration for the Unsharp Mask filtering function from C/C++ library
cdef extern from "../../include/filters/unsharp_mask_filter.h":
    void unsharp_mask_filtering[numeric] (numeric* image, float* output,
                                          int rows, int cols, int depth,
                                          float sigma, float amount, float threshold, bool type)

def unsharp_mask(np.ndarray[numeric, ndim=3] image,
                 np.ndarray[np.float32_t, ndim=3] output,
                 int rows, int cols, int depth,
                 float sigma, float amount, float threshold, bool type):
    """
    Apply an Unsharp Mask filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        sigma (float): Standard deviation for Gaussian kernel.
        amount (float): Amount of the unsharp mask.
        threshold (float): Threshold for the unsharp mask.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return unsharp_mask_filtering(&image[0,0,0],
                                  &output[0,0,0],
                                  rows, cols, depth,
                                  sigma, amount, threshold, type)

# Extern declaration for the Sobel filtering function from C/C++ library
cdef extern from "../../include/filters/sobel_filter.h":
    void sobel_filtering[numeric] (numeric* image, float* output,
                                   int rows, int cols, int depth, bool type)

def sobel(np.ndarray[numeric, ndim=3] image,
          np.ndarray[np.float32_t, ndim=3] output,
          int rows, int cols, int depth, bool type):
    """
    Apply a Sobel filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return sobel_filtering(&image[0,0,0],
                           &output[0,0,0],
                           rows, cols, depth, type)

# Extern declaration for the Prewitt filtering function from C/C++ library
cdef extern from "../../include/filters/prewitt_filter.h":
    void prewitt_filtering[numeric] (numeric* image, float* output,
                                     int rows, int cols, int depth, bool type)

def prewitt(np.ndarray[numeric, ndim=3] image,
            np.ndarray[np.float32_t, ndim=3] output,
            int rows, int cols, int depth, bool type):
    """
    Apply a Prewitt filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        type (bool): Type of filtering (specific to implementation).

    Returns:
        None
    """
    return prewitt_filtering(&image[0,0,0],
                             &output[0,0,0],
                             rows, cols, depth, type)

# Extern declaration for the Canny filtering function from C/C++ library
cdef extern from "../../include/filters/canny_filter.h":
    void canny_filtering[numeric] (numeric* image, float* output,
                                   int rows, int cols, int depth,
                                   float sigma, float low_threshold, float high_threshold)

def canny(np.ndarray[numeric, ndim=3] image,
          np.ndarray[np.float32_t, ndim=3] output,
          int rows, int cols, int depth,
          float sigma, float low_threshold, float high_threshold):
    """
    Apply a Canny filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        sigma (float): Standard deviation for Gaussian kernel.
        low_threshold (float): Lower threshold for the hysteresis procedure.
        high_threshold (float): Upper threshold for the hysteresis procedure.

    Returns:
        None
    """
    return canny_filtering(&image[0,0,0],
                           &output[0,0,0],
                           rows, cols, depth,
                           sigma, low_threshold, high_threshold)


# Extern declaration for the median filtering function from C/C++ library
cdef extern from "../../include/filters/median_filter.h":
    void median_filtering[numeric] (numeric* image, numeric* output,
                                  int rows, int cols, int depth,
                                  int rows_kernel, int cols_kernel, int depth_kernel)

def median(np.ndarray[numeric, ndim=3] image,
         np.ndarray[numeric, ndim=3] output,
         int rows, int cols, int depth,
         int rows_kernel, int cols_kernel, int depth_kernel):
    """
    Apply a median filter to a 3D image.

    Parameters:
        image (np.ndarray[numeric, ndim=3]): Input 3D image array.
        output (np.ndarray[np.float32_t, ndim=3]): Output 3D array to store the filtered result.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.
        depth (int): Number of depth slices in the image.
        rows_kernel (int): Number of rows in the kernel.
        cols_kernel (int): Number of columns in the kernel.
        depth_kernel (int): Number of depth slices in the kernel.

    Returns:
        None
    """
    return median_filtering(&image[0,0,0],
                          &output[0,0,0],
                          rows, cols, depth,
                          rows_kernel, cols_kernel, depth_kernel)

cdef extern from '../../../src/filters/anisotropic_diffusion.h':
    void anisotropicDiffusion2D[dtype](dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols)

    void anisotropicDiffusion3D[dtype](dtype* inputImage, int totalIterations, float deltaT, 
                        float kappa, int diffusionOption, int numRows, int numCols, int numSlices)

    void anisotropicDiffusion2DGPU[dtype](dtype* inputImage, int totalIterations, float deltaT, 
                                float kappa, int diffusionOption, int numRows, int numCols)
    
    void anisotropicDiffusion3DGPU[dtype](dtype* inputImage, int totalIterations, float deltaT, 
                                float kappa, int diffusionOption, int numRows, int numCols, int slices)

def anisotropic_diffusion2D(numeric[:,::1] input_image, int total_iterations,
                          float delta_t, float kappa, int diffusion_option, gpu = False):
    cdef int rows = input_image.shape[0]
    cdef int cols = input_image.shape[1]

    if gpu == True:
        anisotropicDiffusion2DGPU(&input_image[0,0], total_iterations, delta_t, kappa, diffusion_option, rows, cols)
    else:
        anisotropicDiffusion2D(&input_image[0,0], total_iterations, delta_t, kappa, diffusion_option, rows, cols)

    return input_image  # Should the array be returned?

def anisotropic_diffusion3D(numeric[:,:,::1] input_image, int total_iterations,
                          float delta_t, float kappa, int diffusion_option, gpu = False):
    cdef int rows = input_image.shape[0]
    cdef int cols = input_image.shape[1]
    cdef int slices = input_image.shape[2]

    if gpu == True:
        anisotropicDiffusion3DGPU(&input_image[0,0,0], total_iterations, delta_t, kappa, diffusion_option, rows, cols, slices)
    else:
        anisotropicDiffusion3D(&input_image[0,0,0], total_iterations, delta_t, kappa, diffusion_option, rows, cols, slices)

    return input_image  # Should the array be returned?
