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

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.x, isize.y, isize.z), dtype=numpy.float32)

    adaptativeGaussianThresholdChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          sigma, weight, type3d, verbose, ngpus, gpuMemory)
    
    return hostOutput


def meanThreshold(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   int windowSize=1,float weight = 0 ,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.x, isize.y, isize.z), dtype=numpy.float32)

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
                   int windowSize=1,float weight = 0 ,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.x, isize.y, isize.z), dtype=numpy.float32)

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
                   int windowSize=1,float weight = 0 , numeric range  = 1,int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.x, isize.y, isize.z), dtype=numpy.float32)

    nx = windowSize
    ny = windowSize
    nz = windowSize

    sauvolaThresholdChunked(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z,
                          weight, range,type3d, verbose, gpuMemory,ngpus,
                          nx,ny,nz)
    
    return hostOutput