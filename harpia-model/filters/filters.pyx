cimport cython
cimport numpy
import numpy as np
from libcpp cimport bool
from libc.stdint cimport int32_t

cdef extern from "../src/filters/sobel/sobel_filter.h":
    void sobel_filtering(float* image, float* output,
                         int rows, int cols , int depth, bool type)


def sobel(numpy.float32_t[:,:,:] image,
         numpy.float32_t [:,:,:]  output,
         int rows, int cols, int depth, bool type
     ):

     return sobel_filtering(&image[0,0,0],
                            &output[0,0,0],
                            rows, cols, depth, type)


cdef extern from "../src/filters/gaussian/gaussian_filter.h":
    void gaussian_filtering(float* image, float* output,
                         int rows, int cols , int depth,
                         float sigma, bool type)

def gaussian(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          float sigma, bool type):

          return gaussian_filtering(&image[0,0,0],
                                 &output[0,0,0],
                                 rows, cols, depth,
                                 sigma, type)


cdef extern from "../src/filters/unsharp_mask/unsharp_mask_filter.h":
    void unsharp_mask_filtering(float* image, float* output,
                         int rows, int cols , int depth,
                         float sigma, float ammount, float threshold, bool type)

def unsharp_mask(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          float sigma, float ammount, float threshold, bool type):

          return unsharp_mask_filtering(&image[0,0,0],
                                 &output[0,0,0],
                                 rows, cols, depth,
                                 sigma, ammount, threshold, type)


cdef extern from "../src/filters/LoG/LoG_filter.h":
    void log_filtering(float* image, float* output,
                         int rows, int cols , int depth,
                         bool type)

def LoG(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          bool type):

          return log_filtering(&image[0,0,0],
                                 &output[0,0,0],
                                 rows, cols, depth,
                                type)


cdef extern from "../src/filters/mean/mean_filter.h":
    void mean_filtering(float* image, float* output,
                         int rows, int cols , int depth,
                         int rows_kernel, int cols_kernel, int depth_kernel)

def mean(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          int rows_kernel, int cols_kernel, int depth_kernel):

          return mean_filtering(&image[0,0,0],
                                 &output[0,0,0],
                                 rows, cols, depth,
                                 rows_kernel, cols_kernel, depth_kernel)