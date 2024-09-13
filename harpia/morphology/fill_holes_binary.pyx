cimport numpy
from libc.stdint cimport uint16_t, uint32_t

import numpy

ctypedef fused numeric:
    uint16_t
    int
    unsigned int

cdef extern from "../../include/morphology/morphology.h":
    ctypedef enum MorphOp:
        EROSION
        DILATION

#reconstruction by erosion/dilation
cdef extern from "../../include/morphology/reconstruction_binary.h":
    void reconstruction_binary_on_device[dtype](dtype *, dtype *,const int, const int, const int, dtype *, 
                  MorphOp, const int )

def reconstruction_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                          numpy.ndarray[numeric, ndim=3] hostOutput, 
                          int xsize, 
                          int ysize, 
                          int zsize,
                          numpy.ndarray[numeric, ndim=3] hostMask,
                         
                          int operation, 
                          int flag_verbose):

    cdef MorphOp morph_op
    if operation == 0:
        morph_op = EROSION
    elif operation == 1:
        morph_op = DILATION
    else:
        raise ValueError("Invalid operation. Must be 0 (EROSION) or 1 (DILATION).")
    
    return reconstruction_binary_on_device(&hostImage[0,0,0], 
                                           &hostOutput[0,0,0], 
                                           xsize, ysize, zsize, 
                                           &hostMask[0,0,0],   
                                           morph_op, flag_verbose)

#reconstruction by erosion/dilation
cdef extern from "../../include/morphology/fill_holes.h":
    void fill_holes_on_device[dtype](dtype *, dtype *, const int, const int, const int, const int )

def fill_holes(numpy.ndarray[numeric, ndim=3] hostImage, 
                      numpy.ndarray[numeric, ndim=3] hostOutput, 
                      int xsize,
                      int ysize, 
                      int zsize,
                      int flag_verbose):

    return fill_holes_on_device(&hostImage[0,0,0], &hostOutput[0,0,0], xsize, ysize, zsize, flag_verbose)