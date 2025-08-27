cimport cython
cimport numpy as np
from libcpp cimport bool

# Define the fused type for integer types: float, int, unsigned int
ctypedef fused integer:
    int

# Extern declaration for the Local Gaussian Threshold function from C/C++ library
cdef extern from "../../include/watershed/watershed.h":
    void watershed(int* data, int* labels, int rows, int cols, int iterations)
    void hierarchicalWatershed(int* data, int* labels, int rows, int cols, int levels)


def watershed_cpu(np.ndarray[int, ndim=2] data,
                   np.ndarray[int, ndim=2] labels,
                   int rows, int cols, int iterations):

    return watershed(&data[0,0],
                     &labels[0,0],
                     rows, cols, iterations)


def watershed_hierarchical(np.ndarray[int, ndim=2] data,
                   np.ndarray[int, ndim=2] labels,
                   int rows, int cols, int levels):

    return hierarchicalWatershed(&data[0,0],
                     &labels[0,0],
                     rows, cols, levels)


cdef extern from "../../include/watershed/marker_based_watershed.h":
    void union_find_watershed(int* sortedImage, int* labels, int xsize, int ysize)
    void meyers_watershed_2d(int* hostImage, int* markers, int background, int xsize, int ysize)
    void meyers_watershed_3d(int* hostImage, int* markers, int background, int xsize, int ysize, int zsize)


def union_find_watershed_2d(np.ndarray[int, ndim=2] sortedImage,
                            np.ndarray[int, ndim=2] labels,
                            int rows, int cols):
    print("calling uf 2d")

    return union_find_watershed(&sortedImage[0,0],
                                  &labels[0,0],
                                  rows, cols)


def watershed_meyers_2d(np.ndarray[int, ndim=2] hostImage,
                        np.ndarray[int, ndim=2] markers,
                        int background,
                        int xsize, int ysize):
    print("calling meyers 2d")

    return meyers_watershed_2d(&hostImage[0,0],
                                  &markers[0,0],
                                  background,
                                  xsize, ysize)


def watershed_meyers_3d(np.ndarray[int, ndim=3] hostImage,
                        np.ndarray[int, ndim=3] markers,
                        int background,
                        int xsize, int ysize, int zsize):
    print("calling meyers 3d")

    return meyers_watershed_3d(&hostImage[0,0,0],
                                  &markers[0,0,0],
                                  background,
                                  xsize, ysize, zsize)