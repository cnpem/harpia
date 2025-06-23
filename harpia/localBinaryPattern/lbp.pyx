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
    """
    Compute the Local Binary Pattern (LBP) of a 3D image using a CUDA-accelerated implementation.

    Parameters
    ----------
    hostImage : numpy.ndarray
        Input 3D image array of numeric type (float, int, unsigned int) with shape (z, y, x).
    hostOutput : numpy.ndarray, optional
        Preallocated output array of shape (z, y, x) with dtype float32.
        If None, a new array is allocated and returned. Default is None.
    sigma : float, optional
        Unused in current implementation, reserved for future use. Default is 1.
    type3d : int, optional
        Flag to specify 2D or 3D processing mode. Default is 1.
    verbose : int, optional
        Verbosity level for debugging or logging. Default is 0 (no output).
    gpuMemory : float, optional
        Fraction of GPU memory to use (reserved for future use). Default is 0.4.
    ngpus : int, optional
        Number of GPUs to use. -1 means automatic detection. Default is -1.

    Returns
    -------
    numpy.ndarray
        The output LBP image as a 3D float32 array with the same shape as `hostImage`.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    localBinaryPattern(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z)
    
    return hostOutput