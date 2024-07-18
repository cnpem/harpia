cimport cython
cimport numpy
import numpy as np
from libcpp cimport bool

cdef extern from "../../include/quantification/fraction.h":
    void fraction(int* image, int* output, int xsize, int ysize, int zsize)

def compute_fraction(numpy.int32_t[:,:,:] image,
                     numpy.int32_t [:,:,:]  output,
                     int xsize, int ysize, int zsize):
    
    # Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef int* output_ptr = <int*>&output[0,0,0]

    return fraction(image_ptr, output_ptr, xsize, ysize, zsize)


cdef extern from "../../include/quantification/perimeter.h":
    void perimeter(int* image, unsigned int* output, int xsize, int ysize, int zsize)

def compute_perimeter(numpy.int32_t[:,:,:] image,
                      numpy.uint32_t [:,:,:]  output,
                      int xsize, int ysize, int zsize):

    # Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef unsigned int* output_ptr = <unsigned int*>&output[0,0,0]

    return perimeter(image_ptr, output_ptr, xsize, ysize, zsize)



cdef extern from "../../include/quantification/area.h":
    void area(int* image, unsigned int* output, int xsize, int ysize, int zsize, bool type)

def compute_area(numpy.int32_t[:,:,:] image,
                 numpy.uint32_t [:,:,:]  output,
                 int xsize, int ysize, int zsize, bool type):

    # Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef unsigned int* output_ptr = <unsigned int*>&output[0,0,0]

    return area(image_ptr, output_ptr, xsize, ysize, zsize,type)


cdef extern from "../../include/quantification/volume.h":
    void volume(int* image, unsigned int* output, int xsize, int ysize, int zsize)

def compute_volume(numpy.int32_t[:,:,:] image,
                 numpy.uint32_t [:,:,:]  output,
                 int xsize, int ysize, int zsize):

    # Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef unsigned int* output_ptr = <unsigned int*>&output[0,0,0]

    return volume(image_ptr, output_ptr, xsize, ysize, zsize)
