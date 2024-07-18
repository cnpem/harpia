cimport cython
cimport numpy
import numpy as np
from libcpp cimport bool

cdef extern from "../../include/threshold/adaptative_gaussian.h":
    void local_gaussian_threshold(float* image, float* output, int rows, int cols, int depth, float sigma, float weight, bool type)

def local_gaussian(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          int rows, int cols, int depth,
          float sigma, float weight, bool type):

          return local_gaussian_threshold(&image[0,0,0],
                                  &output[0,0,0],
                                  rows, cols, depth,
                                  sigma,weight,type)


cdef extern from "../../include/threshold/adaptative_mean.h":
    void local_mean_threshold(float* image, float* output,float weight,
                         int rows, int cols , int depth,
                         int rows_kernel, int cols_kernel, int depth_kernel)

def local_mean(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          float weight,
          int rows, int cols, int depth,
          int rows_kernel, int cols_kernel, int depth_kernel):

          return local_mean_threshold(&image[0,0,0],
                                  &output[0,0,0],
                                  weight,
                                  rows, cols, depth,
                                  rows_kernel, cols_kernel, depth_kernel)


cdef extern from "../../include/threshold/niblack.h":
    void niblack_threshold(float* image, float* output,float weight,
                         int rows, int cols , int depth,
                         int rows_kernel, int cols_kernel, int depth_kernel)

def niblack(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          float weight,
          int rows, int cols, int depth,
          int rows_kernel, int cols_kernel, int depth_kernel):

          return niblack_threshold(&image[0,0,0],
                                  &output[0,0,0],
                                  weight,
                                  rows, cols, depth,
                                  rows_kernel, cols_kernel, depth_kernel)


cdef extern from "../../include/threshold/sauvola.h":
    void sauvola_threshold(float* image, float* output,float weight, float range,
                         int rows, int cols , int depth,
                         int rows_kernel, int cols_kernel, int depth_kernel)

def sauvola(numpy.float32_t[:,:,:] image,
          numpy.float32_t [:,:,:]  output,
          float weight, float range,
          int rows, int cols, int depth,
          int rows_kernel, int cols_kernel, int depth_kernel):

          return sauvola_threshold(&image[0,0,0],
                                  &output[0,0,0],
                                  weight, range,
                                  rows, cols, depth,
                                  rows_kernel, cols_kernel, depth_kernel)



