from libc.stdint cimport uint32_t, uint16_t
cimport numpy
import numpy

ctypedef fused numeric:
    uint16_t
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_binary.h":
    void _erosion_binary "erosion_binary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _dilation_binary "dilation_binary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _closing_binary "closing_binary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _opening_binary "opening_binary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)

def erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _erosion_binary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _dilation_binary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def closing_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _closing_binary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def opening_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _opening_binary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)


