cimport cython
cimport numpy
import numpy

from libcpp cimport bool
from harpia.common import Size

#Define the fused type for numeric types : float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int

ctypedef fused real:
    float
    double
    
#---------------------------------------------------------------------------------------------------
cdef extern from "../../include/filters/gaussian_filter.h":
    void gaussianFilterChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, 
                                        int zsize, float sigma, int type3d, int verbose, int ngpus, 
                                        float gpuMemory)

cdef extern from "../../include/filters/mean_filter.h":
    void meanFilterChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, 
                                    int zsize, int type3d, int flag_verbose, float gpuMemory, int ngpus, int nx, 
                                    int ny, int nz)

cdef extern from "../../include/filters/log_filter.h":
    void logFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize, int type3d,
                                    int flag_verbose, int ngpus, float gpuMemory)

cdef extern from "../../include/filters/unsharp_mask_filter.h":
    void unsharpMaskChunked[numeric](numeric* image, float* output, int xsize, int ysize, int zsize,
                                     float sigma, float ammount, float threshold,  int type3d, const int verbose, 
                                     int ngpus, const float safetyMargin)

cdef extern from "../../include/filters/sobel_filter.h":
    void sobelFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize, int type3d,
                                    int flag_verbose, int ngpus, float gpuMemory)

cdef extern from "../../include/filters/prewitt_filter.h":
    void prewittFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                       int xsize, int ysize, int zsize, int type3d,
                                       int flag_verbose, int ngpus, float gpuMemory)

cdef extern from '../../include/filters/anisotropic_diffusion.h':
    void anisotropicDiffusion3D[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, 
                                       float deltaT, float kappa, int diffusionOption, int xsize, 
                                       int ysize, int zsize, const int flag_verbose, 
                                       float gpuMemory, int ngpus)

#---------------------------------------------------------------------------------------------------

def gaussianFilter(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   float sigma = 1, int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):

    """
    Apply a 3D Gaussian filter to a volume using chunked GPU processing.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        sigma (float): Standard deviation of the Gaussian kernel.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Filtered 3D image.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    gaussianFilterChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          sigma, type3d, verbose, ngpus, gpuMemory)
    
    return hostOutput

def meanFilter(numpy.ndarray[numeric, ndim=3] hostImage,
               numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
               int windowSize = 3,
               int type3d = 1,
               int verbose = 0,
               float gpuMemory = 0.4,
               int ngpus = -1,
               ):

    """
    Apply a 3D mean (box) filter to a volume using chunked GPU processing.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        windowSize (int): Size of the mean filter kernel in all dimensions.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Filtered 3D image.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    nx = windowSize
    ny = windowSize
    nz = windowSize

    meanFilterChunked(&hostImage[0, 0, 0],
                      &hostOutput[0, 0, 0],
                      isize.y, isize.x, isize.z, type3d,
                      verbose, gpuMemory, ngpus,
                      nx, ny, nz)

    return hostOutput

def logFilter(numpy.ndarray[numeric, ndim=3] hostImage,
              numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
              int type3d = 1,
              int verbose = 0,
              float gpuMemory = 0.4,
              int ngpus = -1):
    
    """
    Apply a Laplacian of Gaussian (LoG) filter to a 3D image using chunked GPU processing.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Filtered 3D image.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)


    logFilterChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.y, isize.x, isize.z,type3d,
                     verbose, ngpus, gpuMemory)

    return hostOutput

def unsharpMaskFilter(numpy.ndarray[numeric, ndim=3] hostImage,
                      numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                      float sigma = 1,
                      float ammount = 1,
                      float threshold = 0,
                      int type3d = 1,
                      int verbose = 0,
                      float gpuMemory = 0.4,
                      int ngpus = -1):
    """
    Apply a 3D unsharp mask filter to enhance image details using GPU processing.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        sigma (float): Gaussian blur sigma for smoothing.
        ammount (float): Sharpening intensity.
        threshold (float): Intensity threshold for applying sharpening.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Sharpened 3D image.
    """
    
    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    unsharpMaskChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.y, isize.x, isize.z,
                     sigma, ammount, threshold, type3d,
                     verbose, ngpus, gpuMemory)

    return hostOutput

def sobelFilter(numpy.ndarray[numeric, ndim=3] hostImage,
              numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
              int type3d = 1,
              int verbose = 0,
              float gpuMemory = 0.4,
              int ngpus = -1):
    """
    Apply a Sobel edge detection filter in 3D using chunked GPU processing.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Filtered 3D image emphasizing edges.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    sobelFilterChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.y, isize.x, isize.z, type3d,
                     verbose, ngpus, gpuMemory)

    return hostOutput

def prewittFilter(numpy.ndarray[numeric, ndim=3] hostImage,
              numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
              int type3d  = 1,
              int verbose = 0,
              float gpuMemory = 0.4,
              int ngpus = -1):

    """
    Apply a Prewitt edge detection filter in 3D using GPU-based chunked processing.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Filtered 3D image emphasizing edges.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)


    prewittFilterChunked(&hostImage[0, 0, 0],
                     &hostOutput[0, 0, 0],
                     isize.y, isize.x, isize.z, type3d,
                     verbose, ngpus, gpuMemory)

    return hostOutput

def anisotropic_diffusion3D(numpy.ndarray[real, ndim=3] hostImage, 
                            numpy.ndarray[real, ndim=3] hostOutput = None, int total_iterations=10, 
                            float delta_t=0.1, float kappa=30, int diffusion_option=3, int flag_verbose=0, 
                            float gpuMemory=0.4, int ngpus = -1):
    """
    Performs anisotropic diffusion on a 3D image.

    This function applies the anisotropic diffusion algorithm to enhance images by reducing noise 
    while preserving edges. It supports three different diffusion options that control the smoothing 
    behavior.

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
          Mbarki, Zouhair, et al. "A new rapid auto-adapting diffusion function for adaptive 
          anisotropic image de-noising and sharply conserved edges." Computers & Mathematics with 
          Applications 74.8 (2017): 1751-1768.
    flag_verbose: int
        Verbose for number of chuncks in execution
    gpuMemmory: bool
        Percentage of memmory occupied in the GPU (if using the gpu function). With cython value, 
        working value is of 0.4 (40%).
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
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)   

    anisotropicDiffusion3D(&hostImage[0,0,0], &hostOutput[0,0,0], total_iterations, delta_t, kappa, 
    diffusion_option, isize.x, isize.y, isize.z, flag_verbose, gpuMemory, ngpus)

    return hostOutput

