cimport numpy

import numpy

from libc.stdint cimport int16_t, uint16_t
from libcpp cimport int

from harpia.common import Size

ctypedef fused numeric:
    int
    unsigned int
    int16_t
    uint16_t

cdef extern from "../../include/morphology/morphology.h":
    ctypedef enum MorphOp:
        EROSION
        DILATION

cdef extern from "../../include/morphology/operations_binary.h":
    void _erosion_binary "erosion_binary" [dtype](dtype*, dtype*, int, int, int, int, int*, int, 
                                                  int, int, float, int)
    void _dilation_binary "dilation_binary" [dtype](dtype*, dtype*, int, int, int, int, int*, int, 
                                                    int, int, float, int)
    void _closing_binary "closing_binary"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, 
                                                 int, float, int)
    void _opening_binary "opening_binary"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, 
                                                 int, float, int)
    void _smooth_binary "smooth_binary"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, 
                                               int, float, int)
    void _geodesic_erosion_binary "geodesic_erosion_binary"[dtype](dtype*, dtype*, dtype*, int, int, 
                                                                   int, int, float, int)
    void _geodesic_dilation_binary "geodesic_dilation_binary"[dtype](dtype*, dtype*, dtype*, int, 
                                                                     int, int, int, float, int)
    void _reconstruction_binary "reconstruction_binary"[dtype](dtype*, dtype*, dtype*, int, int, 
                                                               int, int, MorphOp, int)
    void _fill_holes "fill_holes"[dtype](dtype*, dtype*, int, int, int, int, int, float, int)

def erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary erosion on a 3D image.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element for erosion.
    :type kernel: int[:, :, :]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.2, on pages 638-643.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _erosion_binary(&hostImage[0,0,0], &hostOutput[0,0,0],  isize.x, isize.y, isize.z, 
                    verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput
    
def dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                    numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                    float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary dilation on a 3D image.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element for erosion.
    :type kernel: int[:, :, :]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.2, on pages 638-643.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _dilation_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                     verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput


def closing_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary closing on a 3D image.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element for erosion.
    :type kernel: int[:, :, :]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.3, on pages 644-648.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _closing_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                    verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

def opening_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary openning on a 3D image.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element for erosion.
    :type kernel: int[:, :, :]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.3, on pages 644-648.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _opening_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                    verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

def smooth_binary(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                   numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary smoothing on a 3D image. The smooth operation consists of a 
    sequence of openning and closing operations.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element for erosion.
    :type kernel: int[:, :, :]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation is based on the morphological operations described in "Digital Image 
       Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, particularly in Chapter 9 
       (Morphological Image Processing), Section 9.8, on page 682. The grayscale smoothing 
       algorithm described was adapted for binary images.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _smooth_binary(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                   &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

def geodesic_erosion_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                            numpy.ndarray[numeric, ndim=3] hostMask, 
                            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                            float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary geodesic erosion on a 3D image.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param hostMask: 3D mask image that acts as a constraint, limiting the extent of erosion 
                     on the input image.    
    :type hostMask: numpy.ndarray[numeric, ndim=3]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 667-668.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    
    _geodesic_erosion_binary(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                             isize.y, isize.z, verbose, gpuMemory, ngpus)

    return hostOutput

def geodesic_dilation_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                             numpy.ndarray[numeric, ndim=3] hostMask, 
                             numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                             float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary geodesic dilation on a 3D image.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param hostMask: 3D mask image that acts as a constraint, limiting the extent of dilation 
                     on the input image.    
    :type hostMask: numpy.ndarray[numeric, ndim=3]
    :param hostOutput: Optional output array. If None, a new array is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level.
    :type verbose: int, default 0
    :param gpuMemory: Fraction of GPU memory to use (0-1).
    :type gpuMemory: float, default 0.4
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The eroded image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 667-668.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)

    _geodesic_dilation_binary(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                              isize.y, isize.z, verbose, gpuMemory, ngpus)

    return hostOutput

def reconstruction_binary(numpy.ndarray[numeric, ndim=3] hostImage, 
                          numpy.ndarray[numeric, ndim=3] hostMask, str operation, 
                          numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                          int ngpus = -1):
    """
    Performs morphological reconstruction on a binary 3D image using erosion or dilation.

    :param hostImage: Input 3D binary image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param hostMask: 3D mask image that acts as a constraint, limiting the extent of the operation.
    :type hostMask: numpy.ndarray[numeric, ndim=3]
    :param operation: Morphological operation to perform. Must be either 'erosion' or 'dilation'.
    :type operation: str
    :param hostOutput: Output 3D image. If None, a new array with the same shape as `hostImage` is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param ngpus: Whether to execute on GPU or CPU. 
                  If ngpus = 0, CPU execution is selected. 
                  Otherwise, the function executes on GPU.
    :type ngpus: int, default -1
    :raises ValueError: If `operation` is not 'erosion' or 'dilation'.
    :return: The reconstructed 3D binary image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 668-669.
    """ 

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)

    cdef MorphOp morph_op
    if operation == "erosion":
        morph_op = EROSION
    elif operation == "dilation":
        morph_op = DILATION
    else:
        raise ValueError("Invalid operation. Must be 'erosion' or 'dilation'.")
    
    _reconstruction_binary(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                           isize.y, isize.z, verbose, morph_op, ngpus)

    return hostOutput

def fill_holes(numpy.ndarray[numeric, ndim=3] hostImage, 
               numpy.ndarray[numeric, ndim=3] hostOutput = None,  int padding = 50, int verbose = 0, 
               float gpuMemory = 0.4, int ngpus = -1):
    """
    Fills holes in a binary 3D image.

    :param hostImage: Input 3D binary image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param hostOutput: Output 3D image. If None, a new array with the same shape as `hostImage` is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
    :param padding: Padding size for the operation.
    :type padding: int, optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param gpuMemory: Fraction of available GPU memory to use (between 0 and 1).
    :type gpuMemory: float, optional
    :param ngpus: Whether to execute on GPU or CPU. 
                  If ngpus = 0, CPU execution is selected. 
                  Otherwise, the function executes on GPU.
    :type ngpus: int, default -1
    :return: The processed 3D binary image with holes filled.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 671-672.
    """
        
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    
    _fill_holes(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, padding, verbose, 
                gpuMemory, ngpus)

    return hostOutput
