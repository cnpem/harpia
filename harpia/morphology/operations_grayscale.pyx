from libc.stdint cimport uint32_t
cimport numpy
import numpy

ctypedef fused numeric:
    float
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_grayscale.h":
    void _erosionGrayscale "erosionGrayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _dilationGrayscale "dilationGrayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _closingGrayscale "closingGrayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _openingGrayscale "openingGrayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)

def erosionGrayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _erosionGrayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def dilationGrayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _dilationGrayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def closingGrayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _closingGrayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def openingGrayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _openingGrayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

#top_hat
cdef extern from "../../include/morphology/top_hat.h":
    void _topHat "topHat"[dtype](dtype *, dtype *, int *, int, int, int, const int, const int, const int, const int)

def topHat(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
           const int xsize, const int ysize, const int zsize, const int flag_verbose):
    return  _topHat(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)

#bottom_hat
cdef extern from "../../include/morphology/bottom_hat.h":
    void _bottomHat "bottomHat"[dtype](dtype *, dtype *, int *, int, int, int, const int, const int, const int, const int)

def bottomHat(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
           const int xsize, const int ysize, const int zsize, const int flag_verbose):
    return  _bottomHat(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)             