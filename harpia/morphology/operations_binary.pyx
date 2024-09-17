cimport numpy
import numpy as np
from libc.stdint cimport uint16_t

import numpy

from harpia.common import Size

ctypedef fused numeric:
    uint16_t
    int
    unsigned int

# basic operations
cdef extern from "../../include/morphology/operations_binary.h":
    void erosion_binary_on_device[dtype](dtype *, dtype *, int, int, int, int *, int, int, int,
                                         int)
    void dilation_binary_on_device[dtype](dtype *, dtype *, int, int, int, int *, int, int, int, 
                                          int)
    void closing_binary_on_device[dtype](dtype *, dtype *, int, int, int,int *, int, int, int, int)
    void opening_binary_on_device[dtype](dtype *, dtype *, int, int, int,int *, int, int, int, int)
    void geodesic_erosion_binary_on_device[dtype](dtype *, dtype *, int, int, int, dtype *,int)
    void geodesic_dilation_binary_on_device[dtype](dtype *, dtype *, int, int, int, dtype *, int)

def erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, int flag_verbose,
                   numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)

    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    erosion_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0],  isize.x, isize.y, isize.z, 
                             &kernel[0,0,0], ksize.x, ksize.y, ksize.z, flag_verbose)

    return hostOutput
    
def dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, int flag_verbose, 
                    numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    dilation_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                              &kernel[0,0,0],ksize.x, ksize.y, ksize.z, flag_verbose)

    return hostOutput


def closing_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, int flag_verbose, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    closing_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z,  
                             &kernel[0,0,0], ksize.x, ksize.y, ksize.z, flag_verbose)
    return hostOutput

def opening_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, int flag_verbose, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    opening_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                             &kernel[0,0,0], ksize.x, ksize.y, ksize.z, flag_verbose)
    return hostOutput

def geodesic_erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                            numpy.ndarray[numeric, ndim=3] hostMask, int flag_verbose, 
                            numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)
    
    geodesic_erosion_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, 
                                      isize.z, &hostMask[0,0,0], flag_verbose)
    return hostOutput

def geodesic_dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                             numpy.ndarray[numeric, ndim=3] hostMask, 
                             int flag_verbose, 
                             numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    geodesic_dilation_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, 
                                       isize.z, &hostMask[0,0,0], flag_verbose)
    return hostOutput

