cimport cython
cimport numpy
import numpy as np
from libcpp cimport bool
from libc.stdint cimport int32_t

cdef extern from "../src/morphology/connected_components/ccl.h":
    void connectedComponents(int32_t* image, int32_t* output, int xsize, int ysize)


def labeling(numpy.int32_t[:,:] image,
             numpy.int32_t [:,:]  output,
             int xsize, int ysize
         ):
    return connectedComponents(&image[0,0], &output[0,0], xsize, ysize)


cdef extern from "../src/morphology/remove_islands/remove_islands.h":
    void _remove_islands "remove_islands"(int32_t* image, int32_t* output, int threshold, int xsize, int ysize)


def remove_islands(numpy.int32_t[:,:] image,
             numpy.int32_t [:,:]  output,
             int threshold,
             int xsize, int ysize
            ):
    return _remove_islands(&image[0,0], &output[0,0], threshold, xsize, ysize) 
