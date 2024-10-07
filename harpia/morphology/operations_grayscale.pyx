cimport numpy
import numpy as np

from libcpp cimport bool

import numpy

from harpia.common import Size

ctypedef fused numeric:
    float
    int
    unsigned int

cdef extern from "../../include/morphology/morphology.h":
    ctypedef enum MorphOp:
        EROSION
        DILATION

cdef extern from "../../include/morphology/operations_grayscale.h":
    void _erosion_grayscale "erosion_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                       int, int, int, bool)
    void _dilation_grayscale "dilation_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                         int, int, int, bool)
    void _closing_grayscale "closing_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                       int, int, int, bool)
    void _opening_grayscale "opening_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                       int, int, int, bool)
    void _geodesic_erosion_grayscale "geodesic_erosion_grayscale"[dtype](dtype*, dtype*, dtype*, int, int, 
                                                                   int, int, bool)
    void _geodesic_dilation_grayscale "geodesic_dilation_grayscale"[dtype](dtype*, dtype*, dtype*, int, 
                                                                     int, int, int, bool)
    void _reconstruction_grayscale "reconstruction_grayscale"[dtype](dtype*, dtype*, dtype*, int, int, 
                                                               int, int, MorphOp, bool)
    void _bottom_hat "bottom_hat"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, int, 
                                         bool)       
    void _top_hat "top_hat"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, int, bool)


def erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                      numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                      bool gpu = True):

    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    _erosion_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                       &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpu)

    return hostOutput

def dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                       numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                       bool gpu = True):

    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    _dilation_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                        &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpu)

    return hostOutput

def closing_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                      numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                      bool gpu = True):

    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)
    
    _closing_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose,
                       &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpu)

    return hostOutput

def opening_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                      numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                      bool gpu = True):
    isize = Size(hostImage)
    ksize = Size(kernel)

    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)
    
    _opening_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose,
                       &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpu)

    return hostOutput

def geodesic_erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, 
                            numpy.ndarray[numeric, ndim=3] hostMask, 
                            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                            bool gpu = True):
    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)
    
    _geodesic_erosion_grayscale(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                             isize.y, isize.z, verbose, gpu)

    return hostOutput

def geodesic_dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, 
                             numpy.ndarray[numeric, ndim=3] hostMask, 
                             numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                             bool gpu = True):

    isize = Size(hostImage)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    _geodesic_dilation_grayscale(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                              isize.y, isize.z, verbose, gpu)

    return hostOutput

def reconstruction_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, 
                          numpy.ndarray[numeric, ndim=3] hostMask, int operation, 
                          numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                          bool gpu = True):
    
    isize = Size(hostImage)

    cdef MorphOp morph_op
    if operation == 0:
        morph_op = EROSION
    elif operation == 1:
        morph_op = DILATION
    else:
        raise ValueError("Invalid operation. Must be 0 (EROSION) or 1 (DILATION).")
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    _reconstruction_grayscale(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                           isize.y, isize.z, verbose, morph_op, gpu)

    return hostOutput

def top_hat(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, bool gpu = True):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    _top_hat(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
             &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpu)

    return hostOutput


def bottom_hat(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
               numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, bool gpu = True):
    isize = Size(hostImage)
    ksize = Size(kernel)
    
    if hostOutput is None:
        hostOutput = np.empty_like(hostImage)

    _bottom_hat(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpu)             
    return hostOutput
