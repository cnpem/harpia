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
cdef extern from "../include/filters/gaussian_filter.h":
    void gaussianFilterChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, 
                                        int zsize, float sigma, int type3d, int verbose, int ngpus, 
                                        float gpuMemory)

cdef extern from "../include/filters/mean_filter.h":
    void meanFilterChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, 
                                    int zsize, int type3d, int flag_verbose, float gpuMemory, int ngpus, int nx, 
                                    int ny, int nz)

cdef extern from "../include/filters/log_filter.h":
    void logFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize, int type3d,
                                    int flag_verbose, int ngpus, float gpuMemory)

cdef extern from "../include/filters/unsharp_mask_filter.h":
    void unsharpMaskChunked[numeric](numeric* image, float* output, int xsize, int ysize, int zsize,
                                     float sigma, float ammount, float threshold,  int type3d, const int verbose, 
                                     int ngpus, const float safetyMargin)

cdef extern from "../include/filters/sobel_filter.h":
    void sobelFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                    int xsize, int ysize, int zsize, int type3d,
                                    int flag_verbose, int ngpus, float gpuMemory)

cdef extern from "../include/filters/prewitt_filter.h":
    void prewittFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                       int xsize, int ysize, int zsize, int type3d,
                                       int flag_verbose, int ngpus, float gpuMemory)

cdef extern from '../include/filters/anisotropic_diffusion.h':
    void anisotropicDiffusion3D[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, 
                                       float deltaT, float kappa, int diffusionOption, int xsize, 
                                       int ysize, int zsize, const int flag_verbose, 
                                       float gpuMemory, int ngpus)
#Extern declaration for the median filtering function from C / C++ library
cdef extern from "../include/filters/median_filter.h":
    void median_filtering[numeric] (numeric* hostImage, numeric* hostOutput,
                                  int xsize, int ysize, int zsize,
                                  int nx, int ny, int nz)

cdef extern from "../include/filters/canny_filter.h":
    void canny_filtering[numeric] (numeric* hostImage, float* hostOutput,
                                   int xsize, int ysize, int zsize,
                                   float sigma, float low_threshold, float high_threshold)
#---------------------------------------------------------------------------------------------------

def gaussian(numpy.ndarray[numeric, ndim=3] hostImage,
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

def mean(numpy.ndarray[numeric, ndim=3] hostImage,
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


def median(numpy.ndarray[numeric, ndim=3] hostImage,
           numpy.ndarray[numeric, ndim=3] hostOutput = None,
         int nx = 1, int ny = 1, int nz = 1):
    """
    Apply a median filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3]): hostOutput 3D array to store the filtered result.
        nx (int): Number of rows in the kernel.
        ny (int): Number of columns in the kernel.
        nz (int): Number of slices in the kernel.
    """
    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    median_filtering(&hostImage[0,0,0],
                          &hostOutput[0,0,0],
                          isize.y, isize.x, isize.z,
                          nx, ny, 1)

    return hostOutput

def laplace(numpy.ndarray[numeric, ndim=3] hostImage,
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

def unsharp_mask(numpy.ndarray[numeric, ndim=3] hostImage,
                      numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                      float radius = 1,
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
        radius (float): Equals to  the gaussian blur sigma value for smoothing.
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
                     radius, ammount, threshold, type3d,
                     verbose, ngpus, gpuMemory)

    return hostOutput

def sobel(numpy.ndarray[numeric, ndim=3] hostImage,
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

def prewitt(numpy.ndarray[numeric, ndim=3] hostImage,
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


def canny(numpy.ndarray[numeric, ndim=3] hostImage,
          numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
          float sigma = 1.0, float low_threshold = 0.1, float high_threshold = 0.3):
    """
    Apply a Canny filter to a 3D hostImage.

    Parameters:
        hostImage (np.ndarray[numeric, ndim=3]): Input 3D hostImage array.
        hostOutput (np.ndarray[np.float32_t, ndim=3], optional): Output 3D array to store the filtered result.
        sigma (float): Standard deviation for Gaussian smoothing.
        low_threshold (float): Lower threshold for hysteresis.
        high_threshold (float): Upper threshold for hysteresis.
    """
    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    canny_filtering(&hostImage[0,0,0],
                    &hostOutput[0,0,0],
                    isize.y, isize.x, isize.z,
                    sigma, low_threshold, high_threshold)

    return hostOutput
