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
#---------------------------------------------------------------------------------------------------
cdef extern from "../../include/localBinaryPattern/lbp.h":
    void localBinaryPattern[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, int zsize)

def LBP(numpy.ndarray[numeric, ndim=3] hostImage,
                   numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                   float sigma = 1, int type3d=1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1
                   ):

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    localBinaryPattern(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z)
    
    return hostOutput