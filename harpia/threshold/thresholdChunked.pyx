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
cdef extern from "../../include/threshold/adaptative_gaussian.h":
    void adaptativeGaussianThresholdChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, 
                                        int zsize, float sigma, float weight, int type3d, int verbose, int ngpus, 
                                        float gpuMemory)


cdef extern from "../../include/threshold/adaptative_mean.h":
    void adaptativeMeanThresholdChunked[numeric](numeric* hostImage, float* hostOutput,
        int xsize, int ysize, int zsize, float weight,
        int type3d, int verbose, float gpuMemory, int ngpus,
        int nx, int ny, int nz)


cdef extern from "../../include/threshold/niblack.h":
    void niblackThresholdChunked[numeric](numeric* hostImage, float* hostOutput,
        int xsize, int ysize, int zsize, float weight,
        int type3d, int verbose, float gpuMemory, int ngpus,
        int nx, int ny, int nz)


cdef extern from "../../include/threshold/sauvola.h":
    void sauvolaThresholdChunked[numeric](numeric* hostImage, float* hostOutput,
        int xsize, int ysize, int zsize, float weight, numeric range,
        int type3d, int verbose, float gpuMemory, int ngpus,
        int nx, int ny, int nz)


#---------------------------------------------------------------------------------------------------
def gaussianThreshold(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   float sigma = 1, float weight = 0 ,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):
    """
    Apply adaptive Gaussian thresholding to a 3D image using GPU chunking.

    The threshold is computed by subtracting a weighted Gaussian-filtered value from the input.

    Parameters:
        hostImage (ndarray): Input 3D image of numeric type.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        sigma (float): Standard deviation for Gaussian kernel.
        weight (float): Constant subtracted from the mean (bias).
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Binarized image based on adaptive Gaussian threshold.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    adaptativeGaussianThresholdChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          sigma, weight, type3d, verbose, ngpus, gpuMemory)
    
    return hostOutput


def meanThreshold(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   int windowSize=3,float weight = 0 ,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):
    
    """
    Apply adaptive mean thresholding to a 3D image using GPU chunking.

    The threshold is computed by subtracting a constant `weight` from the local mean in a window.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        windowSize (int): Size of the mean filter window (applied in all directions).
        weight (float): Constant bias subtracted from local mean.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Adaptive mean thresholded binary volume.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    nx = windowSize
    ny = windowSize
    nz = windowSize

    adaptativeMeanThresholdChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          weight, type3d, verbose, gpuMemory,ngpus,
                          nx,ny,nz)
    
    return hostOutput


def niblackThreshold(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   int windowSize=3,float weight = 0 ,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):
    """
    Apply the Niblack thresholding method to a 3D image using GPU chunking.

    The threshold is calculated as:  
    **T = mean + weight × stddev**,  
    where mean and stddev are computed over the local window.

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        windowSize (int): Size of the local window.
        weight (float): Scaling factor for the local standard deviation.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Binary result after Niblack thresholding.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    nx = windowSize
    ny = windowSize
    nz = windowSize

    niblackThresholdChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          weight, type3d, verbose, gpuMemory,ngpus,
                          nx,ny,nz)
    
    return hostOutput


def sauvolaThreshold(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   int windowSize=3,float weight = 0 , numeric range  = 1,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):

    """
    Apply Sauvola thresholding to a 3D image using GPU chunking.

    The threshold is computed as:  
    **T = mean × (1 + weight × (stddev / range - 1))**

    Parameters:
        hostImage (ndarray): Input 3D image.
        hostOutput (ndarray, optional): Output array (float32) to store the result. Auto-created if None.
        windowSize (int): Neighborhood size for local statistics.
        weight (float): Parameter `k` in the Sauvola formula.
        range (numeric): Dynamic range of image intensity values.
        type3d (int): Use full 3D filtering (1) or slice-wise (0).
        verbose (int): Verbose for number of chuncks in execution.
        gpuMemory (float): Fraction of GPU memory to use (0–1).
        ngpus (int): Number of GPUs to utilize (-1 = all available).

    Returns:
        ndarray: Binarized volume after Sauvola adaptive thresholding.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    nx = windowSize
    ny = windowSize
    nz = windowSize

    sauvolaThresholdChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          weight, range,type3d, verbose, gpuMemory,ngpus,
                          nx,ny,nz)
    
    return hostOutput