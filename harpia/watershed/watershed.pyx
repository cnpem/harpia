cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool
from harpia.common import Size

ctypedef fused numeric:
    float
    int
    unsigned int

# Define the fused type for integer types: float, int, unsigned int
ctypedef fused integer:
    int

# Extern declaration for the Watershed functions from C/C++ library
cdef extern from "../../include/watershed/watershed.h":
    void watershed(int* data, int* labels, int rows, int cols, int iterations)
    void hierarchicalWatershed(int* data, int* labels, int rows, int cols, int levels)

    void watershed3d(int* data, int* labels, int rows, int cols, int depth, int iterations)
    void hierarchicalWatershed3d(int* data, int* labels, int rows, int cols, int depth, int levels)

    void hierarchicalWatershed_2d_batched(int* image,
                                      int rows, int cols, int depth,
                                      int* labels,
                                      int levels,
                                      int dz)

    void watershed_gpu[numeric](numeric* h_image, int* h_labels, int rows, int cols)

    void hierarchicalWatershed_gpu[numeric](const numeric* h_image, int* h_labels,
                               int ysize, int xsize, int levels)

    void hierarchicalWatershed_gpu_3d[numeric](const numeric* hostImage, int* hostLabels,
                                  int xsize, int ysize, int zsize, int flag_verbose, int levels,int neighborhood)


    void hierarchicalWatershedChunked[numeric](numeric* hostImage, int* hostLabels,
                                  int xsize, int ysize, int zsize,
                                  int levels, int neighborhood,
                                  float gpuMemory, int ngpus, int flag_verbose)


def hierarchicalWatershedChunked_GPU(np.ndarray[numeric, ndim=3] hostImage,
                                     np.ndarray[int, ndim=3] hostLabels = None,
                                     int levels = 3,
                                     int neighborhood = 26,
                                     float gpuMemory = 0.4,
                                     int ngpus = -1,
                                     int verbose = 1):

    isize = Size(hostImage)

    if hostLabels is None:
        hostLabels = np.empty((isize.z, isize.y, isize.x), dtype=np.int32)

    if verbose:
        print(f"Volume: {isize.z}x{isize.y}x{isize.x}, levels={levels}, neighborhood={neighborhood}")

    hierarchicalWatershedChunked(
        &hostImage[0,0,0],
        &hostLabels[0,0,0],
        isize.x, isize.y, isize.z,
        levels, neighborhood,
        gpuMemory, ngpus, verbose
    )
    return hostLabels


def watershed_GPU(np.ndarray[numeric, ndim=2] image,
                   np.ndarray[int, ndim=2] labels,
                   int rows, int cols):

    return watershed_gpu(&image[0,0],
                     &labels[0,0],
                     rows, cols)


def hierarchicalWatershed_GPU(np.ndarray[numeric, ndim=2] image,
                              np.ndarray[int, ndim=2] labels,
                              int ysize, int xsize, int levels):
    return hierarchicalWatershed_gpu(&image[0,0], &labels[0,0], ysize, xsize, levels)


def hierarchicalWatershed3D_GPU(np.ndarray[numeric, ndim=3] image,
                              np.ndarray[int, ndim=3] labels,
                              int rows, int cols, int depth, int verbose, int levels, int neighborhood):
    return hierarchicalWatershed_gpu_3d(&image[0,0,0], &labels[0,0,0], rows, cols, depth, verbose, levels,neighborhood)

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


def watershed3d_cpu(np.ndarray[int, ndim=3] data,
                    np.ndarray[int, ndim=3] labels,
                    int rows, int cols, int depth, int iterations):

    return watershed3d(&data[0,0,0],
                       &labels[0,0,0],
                       rows, cols, depth, iterations)


def watershed3d_hierarchical(np.ndarray[int, ndim=3] data,
                             np.ndarray[int, ndim=3] labels,
                             int rows, int cols, int depth, int levels):

    return hierarchicalWatershed3d(&data[0,0,0],
                                   &labels[0,0,0],
                                   rows, cols, depth, levels)


def watershed2d_hierarchical_batches(np.ndarray[int, ndim=3] data,
                                     np.ndarray[int, ndim=3] labels,
                                     int rows, int cols, int depth,
                                     int levels, int dz=16):

    hierarchicalWatershed_2d_batched(<int*> data.data,
                                     rows, cols, depth,
                                     <int*> labels.data,
                                     levels, dz)


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