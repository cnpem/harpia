cimport numpy
import numpy as np

from libc.stdint cimport uint32_t

import numpy

from harpia.common import Size

ctypedef fused numeric:
    float
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_grayscale.h":
    void erosion_grayscale_on_device[dtype](dtype *, dtype *, int, int, int, int *, int, int, int, int)
    void dilation_grayscale_on_device[dtype](dtype *, dtype *, int, int, int,int *, int, int, int, int)
    void closing_grayscale_on_device[dtype](dtype *, dtype *, int, int, int, int *, int, int, int, int)
    void opening_grayscale_on_device[dtype](dtype *, dtype *, int, int, int, int *,int, int, int, int)

def erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
                   int flag_verbose, numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    erosion_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, flag_verbose)
    return hostOutput

def dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage,  int[:,:,:] kernel,  
                    int flag_verbose, numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    dilation_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, 
                             flag_verbose)
    return hostOutput

def closing_grayscale(numpy.ndarray[numeric, ndim=3] hostImage,  int[:,:,:] kernel,  
                   int flag_verbose, numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)
    
    closing_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0],  isize.x, isize.y, isize.z, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, 
                            flag_verbose)
    return hostOutput

def opening_grayscale(numpy.ndarray[numeric, ndim=3] hostImage,  int[:,:,:] kernel,  
                   int flag_verbose, numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)

    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)
    
    opening_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z,  &kernel[0,0,0], ksize.x, ksize.y, ksize.z, 
                            flag_verbose)
    return hostOutput

#top_hat
cdef extern from "../../include/morphology/top_hat.h":
    void top_hat_on_device[dtype](dtype *, dtype *, const int, const int, const int,  int *, int, int, int, const int)

def top_hat(numpy.ndarray[numeric, ndim=3] hostImage,  int[:,:,:] kernel,  
           const int flag_verbose, numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    top_hat_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, &kernel[0,0,0], ksize.x, ksize.y, ksize.z,  flag_verbose)
    return hostOutput

#bottom_hat
cdef extern from "../../include/morphology/bottom_hat.h":
    void bottom_hat_on_device[dtype](dtype *, dtype *, const int, const int, const int,  int *, int, int, int, const int)

def bottom_hat(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,  
            const int flag_verbose, numpy.ndarray[numeric, ndim=3] hostOutput = None):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    bottom_hat_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, &kernel[0,0,0], ksize.x, ksize.y, ksize.z,  flag_verbose)             
    return hostOutput
