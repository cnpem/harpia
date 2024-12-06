cimport cython
cimport numpy as np
import numpy as np

from libcpp cimport bool
from harpia.common import Size

#Define the fused type for numeric types : float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int

#Extern declaration for the Gaussian filtering function from C / C++ library
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

#Extern declaration for the Mean filtering function from C / C++ library
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

#Extern declaration for the LoG filtering function from C / C++ library
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

#Extern declaration for the Unsharp Mask filtering function from C / C++ library
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

#Extern declaration for the Sobel filtering function from C / C++ library
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

#Extern declaration for the Prewitt filtering function from C / C++ library
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

#Extern declaration for the Canny filtering function from C / C++ library
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

#Extern declaration for the median filtering function from C / C++ library
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

#Define the fused type for numeric types : float, double
ctypedef fused real:
    float
    double

cdef extern from '../../include/filters/anisotropic_diffusion.h':

    void anisotropicDiffusion2DGPU[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize)                        

    void anisotropicDiffusion3D[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize, int zsize, const int flag_verbose, float gpuMemory, bool gpu)


def anisotropic_diffusion2D(np.ndarray[real, ndim=2] hostImage, int total_iterations,
                          float delta_t, float kappa, int diffusion_option):
    """
    Performs anisotropic diffusion on a 2D image.

    This function applies the anisotropic diffusion algorithm to enhance images by reducing noise while preserving edges.
    It supports three different diffusion options that control the smoothing behavior.

    Parameters:
    -----------
    input_image : float numpy.ndarray
        The input 2D image data.
    total_iterations : int
        Number of iterations to perform.
    delta_t : float
        Time step size.
    kappa : float
        Gradient modulus threshold that influences the conduction.
    diffusion_option : int
        Choice of diffusion function:
        - 1: Exponential decay
        - 2: Inverse quadratic decay
        - 3: Hyperbolic tangent decay
          Option 3 is a faster implementation based on:
          Mbarki, Zouhair, et al. "A new rapid auto-adapting diffusion function for adaptive anisotropic 
          image de-noising and sharply conserved edges." Computers & Mathematics with Applications 74.8 (2017): 1751-1768.

    Returns:
    --------
    output_image
        The diffused image in the same data type as the input..
    """

    # Define variables
    isize = Size(hostImage)
    
    # Create the output array
    cdef np.ndarray[real, ndim=2] hostOutput = np.empty_like(hostImage)

    anisotropicDiffusion2DGPU(&hostImage[0,0], &hostOutput[0,0], total_iterations, delta_t, kappa, diffusion_option, isize.x, isize.y)

    return hostOutput

def anisotropic_diffusion3D(np.ndarray[real, ndim=3] hostImage, int total_iterations,
                          float delta_t, float kappa, int diffusion_option, int flag_verbose, float gpuMemory, bool gpu):
    """
    Performs anisotropic diffusion on a 3D image.

    This function applies the anisotropic diffusion algorithm to enhance images by reducing noise while preserving edges.
    It supports three different diffusion options that control the smoothing behavior.

    Parameters:
    -----------
    input_image : float numpy.ndarray
        The input 3D image data.
    total_iterations : int
        Number of iterations to perform.
    delta_t : float
        Time step size.
    kappa : float
        Gradient modulus threshold that influences the conduction.
    diffusion_option : int
        Choice of diffusion function:
        - 1: Exponential decay
        - 2: Inverse quadratic decay
        - 3: Hyperbolic tangent decay
          Option 3 is a faster implementation based on:
          Mbarki, Zouhair, et al. "A new rapid auto-adapting diffusion function for adaptive anisotropic 
          image de-noising and sharply conserved edges." Computers & Mathematics with Applications 74.8 (2017): 1751-1768.

    Returns:
    --------
    output_image
        The diffused image in the same data type as the input.
    """
    # Define variables
    isize = Size(hostImage)
    
    # Create the output array
    cdef np.ndarray[real, ndim=3] hostOutput = np.empty_like(hostImage)

    anisotropicDiffusion3D(&hostImage[0,0,0], &hostOutput[0,0,0], total_iterations, delta_t, kappa, 
    diffusion_option, isize.x, isize.y, isize.z, flag_verbose, gpuMemory, gpu)

    return hostOutput