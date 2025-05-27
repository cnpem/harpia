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
    """
    gaussian_filtering(&hostImage[0,0,0], &hostOutput[0,0,0],
                        xsize, ysize, zsize,
                        sigma, type)


#chunked version
cdef extern from "../../include/filters/gaussian_filter.h":
    void gaussianFilterChunked[numeric](numeric* hostImage,
                                                    float* hostOutput,
                                                    int xsize, int ysize, int zsize,
                                                    float sigma, int verbose, int ngpus, float gpuMemory)


def gaussianChunked(np.ndarray[numeric, ndim=3] hostImage,
                    np.ndarray[float, ndim=3] hostOutput,
                    int xsize, int ysize, int zsize,
                    float sigma):
    
    cdef int verbose = 1
    cdef int ngpus = 1

    gaussianFilterChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          xsize, ysize, zsize,
                          sigma, verbose, ngpus,0.2)

def gaussianFilter(np.ndarray[numeric, ndim=3] hostImage,
                   np.ndarray[float, ndim=3] hostOutput,
                   float sigma = 1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):


    isize = Size(hostImage)
    gaussianFilterChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.x, isize.y, isize.z,
                          sigma, verbose, ngpus, gpuMemory)


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
    """
    mean_filtering(&hostImage[0,0,0],
                   &hostOutput[0,0,0],
                   xsize, ysize, zsize,
                   nx, ny, nz)

#Extern declaration for the Mean filtering with the chunked executor function
cdef extern from "../../include/filters/mean_filter.h":
    void meanFilterChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, int zsize, int flag_verbose,
                       float gpuMemory, int ngpus, int nx, int ny, int nz)

def meanChunked(np.ndarray[numeric, ndim=3] hostImage,
         np.ndarray[np.float32_t, ndim=3] hostOutput,
         int xsize, int ysize, int zsize,
         int nx, int ny, int nz):

        flag_verbose = 1
        gpuMemory = 0.2
        ngpus = 1
        meanFilterChunked(&hostImage[0,0,0],
                   &hostOutput[0,0,0],
                   xsize, ysize, zsize,flag_verbose,gpuMemory,
                   ngpus,nx,ny,nz)


def meanFilter(np.ndarray[numeric, ndim=3] hostImage,
               np.ndarray[np.float32_t, ndim=3] hostOutput,
               int windowSize = 1,
               int verbose = 0,
               float gpuMemory = 0.4,
               int ngpus = -1,
               ):
    
    isize = Size(hostImage)
    nx = windowSize
    ny = windowSize
    nz = windowSize

    meanFilterChunked(&hostImage[0, 0, 0],
                      &hostOutput[0, 0, 0],
                      isize.x, isize.y, isize.z,
                      verbose, gpuMemory, ngpus,
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

    """
    log_filtering(&hostImage[0,0,0],
                  &hostOutput[0,0,0],
                  xsize, ysize, zsize,
                  type)

#chunked version
cdef extern from "../../include/filters/log_filter.h":
    void logFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize,
                                    int flag_verbose, int ngpus, float gpuMemory)

def logChunked(np.ndarray[numeric, ndim=3] hostImage,
               np.ndarray[np.float32_t, ndim=3] hostOutput,
               int xsize, int ysize, int zsize):
    
    cdef int flag_verbose = 1
    cdef int ngpus = 1

    logFilterChunked(&hostImage[0,0,0],
                     &hostOutput[0,0,0],
                     xsize, ysize, zsize,
                     flag_verbose, ngpus, 0.2)


def logFilter(np.ndarray[numeric, ndim=3] hostImage,
              np.ndarray[np.float32_t, ndim=3] hostOutput,
              int verbose = 0,
              float gpuMemory = 0.4,
              int ngpus = -1):

    isize = Size(hostImage)

    logFilterChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.x, isize.y, isize.z,
                     verbose, ngpus, gpuMemory)

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
    """
    unsharp_mask_filtering(&hostImage[0,0,0],
                           &hostOutput[0,0,0],
                           xsize, ysize, zsize,
                           sigma, amount, threshold, type)

#Chunked version
cdef extern from "../../include/filters/unsharp_mask_filter.h":
    void unsharpMaskChunked[numeric](numeric* image, float* output, int xsize, int ysize, int zsize,
                            float sigma, float ammount, float threshold, const int verbose, int ngpus,
                            const float safetyMargin)

def unsharpMaskFilter(np.ndarray[numeric, ndim=3] hostImage,
                      np.ndarray[np.float32_t, ndim=3] hostOutput,
                      float sigma = 1,
                      float ammount = 1,
                      float threshold = 0,
                      int verbose = 0,
                      float gpuMemory = 0.4,
                      int ngpus = -1):
    
    isize = Size(hostImage)
    unsharpMaskChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.x, isize.y, isize.z,
                     sigma, ammount, threshold,
                     verbose, ngpus, gpuMemory)

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
    """
    sobel_filtering(&hostImage[0,0,0],
                    &hostOutput[0,0,0],
                    xsize, ysize, zsize, type)


#Chunked version

cdef extern from "../../include/filters/sobel_filter.h":
    void sobelFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize,
                                    int flag_verbose, int ngpus, float gpuMemory)
                                    
def sobelFilter(np.ndarray[numeric, ndim=3] hostImage,
              np.ndarray[np.float32_t, ndim=3] hostOutput,
              int verbose = 0,
              float gpuMemory = 0.4,
              int ngpus = -1):

    isize = Size(hostImage)

    sobelFilterChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.x, isize.y, isize.z,
                     verbose, ngpus, gpuMemory)

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
    """
    prewitt_filtering(&hostImage[0,0,0],
                      &hostOutput[0,0,0],
                      xsize, ysize, zsize, type)


#Chunked version

cdef extern from "../../include/filters/prewitt_filter.h":
    void prewittFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize,
                                    int flag_verbose, int ngpus, float gpuMemory)

def prewittFilter(np.ndarray[numeric, ndim=3] hostImage,
              np.ndarray[np.float32_t, ndim=3] hostOutput,
              int verbose = 0,
              float gpuMemory = 0.4,
              int ngpus = -1):

    isize = Size(hostImage)

    prewittFilterChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.x, isize.y, isize.z,
                     verbose, ngpus, gpuMemory)

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
    """
    canny_filtering(&hostImage[0,0,0],
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
    """
    return median_filtering(&hostImage[0,0,0],
                          &hostOutput[0,0,0],
                          xsize, ysize, zsize,
                          nx, ny, nz)


#Median filter chunked format (not actually chunked, just in the format of)
def medianFilter(np.ndarray[numeric, ndim=3] hostImage,
               np.ndarray[numeric, ndim=3] hostOutput,
               int nx = 1,
               int ny = 1,
               int nz = 1,
               int verbose = 0,
               float gpuMemory = 0.4,
               int ngpus = -1,
               ):
    
    isize = Size(hostImage)

    median_filtering(&hostImage[0, 0, 0],
                      &hostOutput[0, 0, 0],
                      isize.x, isize.y, isize.z,
                      nx, ny, nz)
#Define the fused type for numeric types : float, double
ctypedef fused real:
    float
    double

cdef extern from '../../include/filters/anisotropic_diffusion.h':

    void anisotropicDiffusion2DGPU[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize)                        

    void anisotropicDiffusion3D[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize, int zsize, const int flag_verbose, float gpuMemory, int ngpus)


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
                          float delta_t, float kappa, int diffusion_option, int flag_verbose, float gpuMemory, int ngpus = -1):
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
    flag_verbose: int
        Verbose for number of chuncks in execution
    gpuMemmory: bool
        Percentage of memmory occupied in the GPU (if using the gpu function). With cython value, working value is of 0.4 (40%).
    ngpus: int 
        The number of GPUs to use. 
        If ngpus < 1, all available GPUs are used.
        If ngpus = 0, CPU execution is selected. 
        If ngpus >= 1, the function uses up to min(ngpus, available GPUs).

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
    diffusion_option, isize.x, isize.y, isize.z, flag_verbose, gpuMemory, ngpus)

    return hostOutput

# Extern declaration for the non-local means filtering function from C/C++ library
cdef extern from "../../include/filters/non_local_means.h":
    void nlmeans_filtering[numeric] (numeric* hostImage, double* hostOutput,
                                     int xsize, int ysize,
                                     int small_window, int big_window, double h, double sigma)

def non_local_means(np.ndarray[numeric, ndim=2] hostImage,
                    np.ndarray[np.float64_t, ndim=2] hostOutput,
                    int xsize, int ysize,
                    int small_window, int big_window, double h, double sigma = 0):
    """
    Apply a non-local means filter to a 2D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=2]): Input 2D hostImage array.
        hostOutput (np.ndarray[np.float64_t, ndim=2]): hostOutput 2D array to store the filtered result.
        xsize (int): Number of rows in the hostImage.
        ysize (int): Number of columns in the hostImage.
        small_window (int): Size of the small window for patch comparison.
        big_window (int): Size of the big window for neighborhood search.
        h (double): Filter parameter for controlling the degree of smoothing.
    """
    nlmeans_filtering(&hostImage[0,0],
                      &hostOutput[0,0],
                      xsize, ysize,
                      small_window, big_window, h, sigma)


def nonLocalMeansFilter(np.ndarray[numeric, ndim=3] hostImage,
                        np.ndarray[np.float64_t, ndim=3] hostOutput,
                        int small_window, int big_window, double h, double sigma=0, int axis=0,
                        int verbose=1, float gpuMemory=1, int ngpus=-1):
    cdef int i
    cdef int xsize, ysize
    cdef int zsize = hostImage.shape[0]
    cdef int ydim = hostImage.shape[1]
    cdef int xdim = hostImage.shape[2]

    if axis == 0:  # slices along z axis, shape (z, y, x) → filter on (y, x)
        xsize, ysize = xdim, ydim
        for i in range(zsize):
            non_local_means(hostImage[i], hostOutput[i], xsize, ysize,
                            small_window, big_window, h, sigma)

    elif axis == 1:  # slices along y axis, shape (z, y, x) → filter on (z, x)
        xsize, ysize = xdim, zsize
        for i in range(ydim):
            non_local_means(hostImage[:, i, :], hostOutput[:, i, :], xsize, ysize,
                            small_window, big_window, h, sigma)

    elif axis == 2:  # slices along x axis, shape (z, y, x) → filter on (z, y)
        xsize, ysize = ydim, zsize
        for i in range(xdim):
            non_local_means(hostImage[:, :, i], hostOutput[:, :, i], xsize, ysize,
                            small_window, big_window, h, sigma)

    else:
        raise ValueError("Axis must be 0, 1, or 2.")
