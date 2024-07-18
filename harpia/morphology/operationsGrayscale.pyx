from libc.stdint cimport uint32_t

cdef extern from "../../include/morphology/operationsGrayscale.h":
    void erosionGrayscale[T] (T *, T *, int *, int, int, int, int, int, int, int, int, int, int)
    void dilationGrayscale[T] (T *, T *, int *, int, int, int, int, int, int, int, int, int, int)
    void closingGrayscale[T] (T *, T *, int *, int, int, int, int, int, int, int, int, int, int)
    void openingGrayscale[T] (T *, T *, int *, int, int, int, int, int, int, int, int, int, int)


ctypedef fused dtype:
    int
    uint32_t
    float


def erosion_grayscale(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
    if dtype is int:
        erosionGrayscale[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is uint32_t:
        erosionGrayscale[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                     xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is float:
        erosionGrayscale[float] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                       xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    else:
        raise ValueError("Unsupported dtype. The supported types are: int32, uint32 and float32")


def dilation_grayscale(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
    if dtype is int:
        dilationGrayscale[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is uint32_t:
        dilationGrayscale[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                     xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is float:
        dilationGrayscale[float] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                       xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    else:
        raise ValueError("Unsupported dtype. The supported types are: int32, uint32 and float32")


def closing_grayscale(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
    if dtype is int:
        closingGrayscale[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is uint32_t:
        closingGrayscale[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                     xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is float:
        closingGrayscale[float] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                       xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    else:
        raise ValueError("Unsupported dtype. The supported types are: int32, uint32 and float32")


def opening_grayscale(dtype[:,:,:] hostImage, dtype[:,:,:] hostOutput, int[:,:,:] kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize, 
                   int xsize, int ysize, int zsize, int block_xsize, int block_ysize, int block_zsize, int flag_verbose):
    if dtype is int:
        openingGrayscale[int] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                            xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is uint32_t:
        openingGrayscale[uint32_t] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                     xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    elif dtype is float:
        openingGrayscale[float] (&hostImage[0,0,0], &hostOutput[0,0,0], &kernel[0,0,0], kernel_xsize, kernel_ysize, kernel_zsize, 
                                       xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)
    else:
        raise ValueError("Unsupported dtype. The supported types are: int32, uint32 and float32")
