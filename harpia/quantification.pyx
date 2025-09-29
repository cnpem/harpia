cimport cython
cimport numpy

import numpy as np
from libcpp cimport bool


#Extern declaration for the fraction calculation function from C / C++ library
cdef extern from "../include/quantification/fraction.h":
    void fraction(int* image, int* output, int xsize, int ysize, int zsize)

cdef extern from "../include/quantification/perimeter.h":
    void perimeter(int* image, unsigned int* output, int xsize, int ysize, int zsize)

cdef extern from "../include/quantification/area.h":
    void area(int* image, unsigned int* output, int xsize, int ysize, int zsize, bool type)

cdef extern from "../include/quantification/volume.h":
    void volume(int* image, unsigned int* output, int xsize, int ysize, int zsize)

cdef extern from "../include/quantification/connected_components.h":
    void connectedComponents(int* image, int* output, int xsize, int ysize, int zsize, bool type)

cdef extern from "../include/quantification/remove_islands.h":
    void remove_islands(int* image, int* output, int threshold, int xsize, int ysize, int zsize, bool type)


def compute_fraction(numpy.int32_t[:,:,:] image,
                     numpy.int32_t[:,:,:] output,
                     int xsize, int ysize, int zsize):
    """
    Compute the fraction of a 3D image.

    Parameters:
        image (numpy.int32_t[:,:,:]): Input 3D image array.
        output (numpy.int32_t[:,:,:]): Output 3D array to store the fraction result.
        xsize (int): Size of the x-dimension.
        ysize (int): Size of the y-dimension.
        zsize (int): Size of the z-dimension.

    Returns:
        None
    """
#Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef int* output_ptr = <int*>&output[0,0,0]

    return fraction(image_ptr, output_ptr, xsize, ysize, zsize)



def compute_perimeter(numpy.int32_t[:,:,:] image,
                      numpy.uint32_t[:,:,:] output,
                      int xsize, int ysize, int zsize):
    """
    Compute the perimeter of a 3D image.

    Parameters:
        image (numpy.int32_t[:,:,:]): Input 3D image array.
        output (numpy.uint32_t[:,:,:]): Output 3D array to store the perimeter result.
        xsize (int): Size of the x-dimension.
        ysize (int): Size of the y-dimension.
        zsize (int): Size of the z-dimension.

    Returns:
        None
    """
#Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef unsigned int* output_ptr = <unsigned int*>&output[0,0,0]

    return perimeter(image_ptr, output_ptr, xsize, ysize, zsize)



def compute_area(numpy.int32_t[:,:,:] image,
                 numpy.uint32_t[:,:,:] output,
                 int xsize, int ysize, int zsize, bool type):
    """
    Compute the area of a 3D image.

    Parameters:
        image (numpy.int32_t[:,:,:]): Input 3D image array.
        output (numpy.uint32_t[:,:,:]): Output 3D array to store the area result.
        xsize (int): Size of the x-dimension.
        ysize (int): Size of the y-dimension.
        zsize (int): Size of the z-dimension.
        type (bool): Type parameter for the area calculation.

    Returns:
        None
    """
#Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef unsigned int* output_ptr = <unsigned int*>&output[0,0,0]

    return area(image_ptr, output_ptr, xsize, ysize, zsize, type)


def compute_volume(numpy.int32_t[:,:,:] image,
                   numpy.uint32_t[:,:,:] output,
                   int xsize, int ysize, int zsize):
    """
    Compute the volume of a 3D image.

    Parameters:
        image (numpy.int32_t[:,:,:]): Input 3D image array.
        output (numpy.uint32_t[:,:,:]): Output 3D array to store the volume result.
        xsize (int): Size of the x-dimension.
        ysize (int): Size of the y-dimension.
        zsize (int): Size of the z-dimension.

    Returns:
        None
    """
#Get pointers to the data
    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef unsigned int* output_ptr = <unsigned int*>&output[0,0,0]

    return volume(image_ptr, output_ptr, xsize, ysize, zsize)


def labelling(numpy.int32_t[:,:,:] image,
                   numpy.int32_t[:,:,:] output,
                   int xsize, int ysize, int zsize):

    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef int* output_ptr = <int*>&output[0,0,0]

    return connectedComponents(image_ptr, output_ptr, xsize, ysize, zsize,type)



def removeIslands(numpy.int32_t[:,:,:] image,
                   numpy.int32_t[:,:,:] output,
                   int threshold,
                   int xsize, int ysize, int zsize):

    cdef int* image_ptr = <int*>&image[0,0,0]
    cdef int* output_ptr = <int*>&output[0,0,0]

    return remove_islands(image_ptr, output_ptr, threshold, xsize, ysize, zsize,type)