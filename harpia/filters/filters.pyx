cimport cython
cimport numpy
import numpy as np
from libcpp cimport bool

cdef extern from "../../include/filters/gaussian_filter.h":
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


cdef extern from "../../include/filters/mean_filter.h":
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

cdef extern from "../../include/filters/log_filter.h":
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


cdef extern from "../../include/filters/unsharp_mask_filter.h":
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


cdef extern from "../../include/filters/sobel_filter.h":
    void sobel_filtering(float* image, float* output,
                         int rows, int cols , int depth, bool type)


def sobel(numpy.float32_t[:,:,:] image,
         numpy.float32_t [:,:,:]  output,
         int rows, int cols, int depth, bool type
     ):

     return sobel_filtering(&image[0,0,0],
                            &output[0,0,0],
                            rows, cols, depth, type)


cdef extern from "../../include/filters/prewitt_filter.h":
    void prewitt_filtering(float* image, float* output,
                         int rows, int cols , int depth, bool type)


def prewitt(numpy.float32_t[:,:,:] image,
         numpy.float32_t [:,:,:]  output,
         int rows, int cols, int depth, bool type
     ):

     return prewitt_filtering(&image[0,0,0],
                            &output[0,0,0],
                            rows, cols, depth, type)


cdef extern from "../../include/filters/deriche_filter.h":
    void deriche_filtering(float* image, float* output,
                         int rows, int cols , int depth,
                         int rows_kernel, int cols_kernel, 
                         float alpha, float low_threshold, float high_threshold)

def deriche(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          int rows_kernel, int cols_kernel, 
          float alpha, float low_threshold, float high_threshold):

          return deriche_filtering(&image[0,0,0],
                                 &output[0,0,0],
                                 rows, cols, depth,
                                 rows_kernel, cols_kernel,
                                 alpha, low_threshold, high_threshold)


cdef extern from "../../include/filters/canny_filter.h":
    void canny_filtering(float* image, float* output,
                         int rows, int cols , int depth,
                         float sigma, float low_threshold, float high_threshold)

def canny(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          float sigma, float low_threshold,float high_threshold):

          return canny_filtering(&image[0,0,0],
                                 &output[0,0,0],
                                 rows, cols, depth,
                                 sigma, low_threshold, high_threshold)