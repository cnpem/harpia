from libc.stdint cimport uint32_t
cimport numpy
import numpy

ctypedef fused numeric:
    float
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_grayscale.h":
    void _erosion_grayscale "erosion_grayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _dilation_grayscale "dilation_grayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _closing_grayscale "closing_grayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _opening_grayscale "opening_grayscale"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)

def erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _erosion_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _dilation_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def closing_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _closing_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def opening_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _opening_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

#top_hat
cdef extern from "../../include/morphology/top_hat.h":
    void _top_hat "top_hat"[dtype](dtype *, dtype *, int *, int, int, int, const int, const int, const int, const int)

def top_hat(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
           const int xsize, const int ysize, const int zsize, const int flag_verbose):
    return  _top_hat(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)

#bottom_hat
cdef extern from "../../include/morphology/bottom_hat.h":
    void _bottom_hat "bottom_hat"[dtype](dtype *, dtype *, int *, int, int, int, const int, const int, const int, const int)

def bottom_hat(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
           const int xsize, const int ysize, const int zsize, const int flag_verbose):
    return  _bottom_hat(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)             