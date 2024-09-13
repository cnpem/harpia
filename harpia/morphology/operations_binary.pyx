cimport numpy
from libc.stdint cimport uint16_t, uint32_t

import numpy

ctypedef fused numeric:
    uint16_t
    int
    unsigned int

#basic operations
cdef extern from "../../include/morphology/operations_binary.h":
    void erosion_binary_on_device[dtype](dtype *, dtype *, int, int, int, int *, int, int, int, int)
    void dilation_binary_on_device[dtype](dtype *, dtype *, int, int, int, int *, int, int, int, int)
    void closing_binary_on_device[dtype](dtype *, dtype *, int, int, int,int *,  int, int, int, int)
    void opening_binary_on_device[dtype](dtype *, dtype *, int, int, int,int *,  int, int, int, int)
    void geodesic_erosion_binary_on_device[dtype](dtype *, dtype *, int, int, int,  dtype *,int)
    void geodesic_dilation_binary_on_device[dtype](dtype *, dtype *, int, int, int,  dtype *, int)

def erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int xsize, int ysize, int zsize,  int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int flag_verbose):
    return erosion_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0],  xsize, ysize, zsize, &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            flag_verbose)

def dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int xsize, int ysize, int zsize, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                    int flag_verbose):
    return dilation_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], xsize, ysize, zsize, &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                             flag_verbose)

def closing_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput,int xsize, int ysize, int zsize, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                    int flag_verbose):
    return closing_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], xsize, ysize, zsize,  &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            flag_verbose)

def opening_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int xsize, int ysize, int zsize, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                    int flag_verbose):
    return opening_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], xsize, ysize, zsize,&kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                             flag_verbose)

def geodesic_erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int xsize, int ysize, int zsize, 
                             numpy.ndarray[numeric, ndim=3] hostMask, int flag_verbose):
    return geodesic_erosion_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], xsize, ysize, zsize,  &hostMask[0,0,0],
                                             flag_verbose)

def geodesic_dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int xsize, int ysize, int zsize, 
                             numpy.ndarray[numeric, ndim=3] hostMask, int flag_verbose):
    return geodesic_dilation_binary_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], xsize, ysize, zsize,  &hostMask[0,0,0], 
                                              flag_verbose)

