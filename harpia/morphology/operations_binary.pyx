from libc.stdint cimport uint32_t, uint16_t
cimport numpy
import numpy

ctypedef fused numeric:
    uint16_t
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_binary.h":
    void _erosionBinary "erosionBinary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _dilationBinary "dilationBinary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _closingBinary "closingBinary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)
    void _openingBinary "openingBinary"[dtype](dtype *, dtype *, int *, int, int, int, int, int, int, int)

def erosionBinary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _erosionBinary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def dilationBinary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _dilationBinary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def closingBinary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _closingBinary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)

def openingBinary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int flag_verbose):
    return _openingBinary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, flag_verbose)


