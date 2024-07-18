from libc.stdint cimport uint16_t, uint32_t
cimport numpy
import numpy

cdef extern from "../../include/morphology/operationsBinary.h":
    void erosionBinary[dtype] (dtype *, dtype *, int *, int, int, int, int, int, int, int, int, int, int)
    void dilationBinary[dtype] (dtype *, dtype *, int *, int, int, int, int, int, int, int, int, int, int)
    void closingBinary[dtype] (dtype *, dtype *, int *, int, int, int, int, int, int, int, int, int, int)
    void openingBinary[dtype] (dtype *, dtype *, int *, int, int, int, int, int, int, int, int, int, int)


ctypedef fused numeric:
    int
    uint32_t
    uint16_t


def erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, numpy.ndarray[numeric, ndim=3] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
    return erosionBinary(&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    

# def dilation_binary(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
#                    int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
#     if dtype is int:
#         dilationBinary[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                             xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     elif dtype is uint32_t:
#         dilationBinary[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                                      xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     elif dtype is uint16_t:
#         dilationBinary[uint16_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     else:
#         raise ValueError("Unsupported dtype")

# def closing_binary(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
#                    int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
#     if dtype is int:
#         closingBinary[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                             xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     elif dtype is uint32_t:
#         closingBinary[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                                      xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     elif dtype is uint16_t:
#         closingBinary[uint16_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     else:
#         raise ValueError("Unsupported dtype")

# def opening_binary(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
#                    int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
#     if dtype is int:
#         openingBinary[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                             xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     elif dtype is uint32_t:
#         openingBinary[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                                      xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     elif dtype is uint16_t:
#         openingBinary[uint16_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
#                                        xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
#     else:
#         raise ValueError("Unsupported dtype")