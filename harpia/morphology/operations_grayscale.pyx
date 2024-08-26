cimport numpy
from libc.stdint cimport uint32_t

import numpy

ctypedef fused numeric:
    float
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_grayscale.h":
    void erosion_grayscale_on_device[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void dilation_grayscale_on_device[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void closing_grayscale_on_device[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void opening_grayscale_on_device[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)

def erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return erosion_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return dilation_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def closing_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return closing_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def opening_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return opening_grayscale_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

#top_hat
cdef extern from "../../include/morphology/top_hat.h":
    void top_hat_on_device[dtype](dtype *, dtype *, int *, int, int, int, const int, const int, const int, const int)

def top_hat(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
           const int xsize, const int ysize, const int zsize, const int flag_verbose):
    return  top_hat_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)

#bottom_hat
cdef extern from "../../include/morphology/bottom_hat.h":
    void bottom_hat_on_device[dtype](dtype *, dtype *, int *, int, int, int, const int, const int, const int, const int)

def bottom_hat(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
           const int xsize, const int ysize, const int zsize, const int flag_verbose):
    return  bottom_hat_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)             