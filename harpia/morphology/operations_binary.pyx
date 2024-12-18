cimport numpy

import numpy

from libc.stdint cimport int8_t, int16_t, uint8_t, uint16_t
from libcpp cimport bool

from harpia.common import Size, ensure_contiguous

ctypedef fused numeric:
    int
    unsigned int
    int16_t
    uint16_t
    int8_t
    uint8_t

cdef extern from "../../include/morphology/morphology.h":
    ctypedef enum MorphOp:
        EROSION
        DILATION

cdef extern from "../../include/morphology/operations_binary.h":
    void _erosion_binary "erosion_binary" [dtype](dtype*, dtype*, int, int, int, int, int*, int, 
                                                  int, int, float, bool)
    void _dilation_binary "dilation_binary" [dtype](dtype*, dtype*, int, int, int, int, int*, int, 
                                                    int, int, float, bool)
    void _closing_binary "closing_binary"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, 
                                                 int, float, bool)
    void _opening_binary "opening_binary"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, 
                                                 int, float, bool)
    void _smooth_binary "smooth_binary"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, 
                                               int, float, bool)
    void _geodesic_erosion_binary "geodesic_erosion_binary"[dtype](dtype*, dtype*, dtype*, int, int, 
                                                                   int, int, float, bool)
    void _geodesic_dilation_binary "geodesic_dilation_binary"[dtype](dtype*, dtype*, dtype*, int, 
                                                                     int, int, int, float, bool)
    void _reconstruction_binary "reconstruction_binary"[dtype](dtype*, dtype*, dtype*, int, int, 
                                                               int, int, MorphOp, bool)
    void _fill_holes "fill_holes"[dtype](dtype*, dtype*, int, int, int, int, bool)

def erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.9, bool gpu = True):

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _erosion_binary(&hostImage[0,0,0], &hostOutput[0,0,0],  isize.x, isize.y, isize.z, 
                    verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, gpu)

    return hostOutput
    
def dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                    numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                    float gpuMemory = 0.9, bool gpu = True):
  
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _dilation_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                     verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, gpu)

    return hostOutput


def closing_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.9, bool gpu = True):

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _closing_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                    verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, gpu)

    return hostOutput

def opening_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.9, bool gpu = True):
  
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _opening_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                    verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, gpu)

    return hostOutput

def smooth_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.9, bool gpu = True):

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _smooth_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                   &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, gpu)

    return hostOutput

def geodesic_erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                            numpy.ndarray[numeric, ndim=3] hostMask, 
                            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                            float gpuMemory = 0.9, bool gpu = True):
 
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    
    _geodesic_erosion_binary(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                             isize.y, isize.z, verbose, gpuMemory, gpu)

    return hostOutput

def geodesic_dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                             numpy.ndarray[numeric, ndim=3] hostMask, 
                             numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                             float gpuMemory = 0.9, bool gpu = True):
   
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)

    _geodesic_dilation_binary(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                              isize.y, isize.z, verbose, gpuMemory, gpu)

    return hostOutput

def reconstruction_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                          numpy.ndarray[numeric, ndim=3] hostMask, str operation, 
                          numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                          bool gpu = True):
      
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)

    cdef MorphOp morph_op
    if operation == "erosion":
        morph_op = EROSION
    elif operation == "dilation":
        morph_op = DILATION
    else:
        raise ValueError("Invalid operation. Must be 'erosion' or 'dilation'.")
    
    _reconstruction_binary(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                           isize.y, isize.z, verbose, morph_op, gpu)

    return hostOutput

def fill_holes(numpy.ndarray[numeric, ndim=3] hostImage, 
               numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, bool gpu = True):
    
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    
    _fill_holes(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, gpu)

    return hostOutput
