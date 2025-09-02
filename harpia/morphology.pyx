cimport numpy

import numpy

from libc.stdint cimport int16_t, uint16_t
from libcpp cimport int
from cython cimport boundscheck, wraparound, parallel

from harpia.common import Size


cdef extern from "../include/morphology/morphology.h":
    ctypedef enum MorphOp:
        EROSION
        DILATION

####################################################################################################
#--------------------------------GRAYSCALE OPERATIONS-----------------------------------------------
####################################################################################################

ctypedef fused numeric:
    float
    int
    unsigned int

cdef extern from "../include/morphology/operations_grayscale.h":
    void _erosion_grayscale "erosion_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                       int, int, int, float, int)
    void _dilation_grayscale "dilation_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                         int, int, int, float, int)
    void _closing_grayscale "closing_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                       int, int, int, float, int)
    void _opening_grayscale "opening_grayscale"[dtype](dtype*, dtype*, int, int, int, int, int*, 
                                                       int, int, int, float, int)
    void _geodesic_erosion_grayscale "geodesic_erosion_grayscale"[dtype](dtype*, dtype*, dtype*, 
                                                                         int, int, int, int, float, 
                                                                         int)
    void _geodesic_dilation_grayscale "geodesic_dilation_grayscale"[dtype](dtype*, dtype*, dtype*, 
                                                                           int, int, int, int, 
                                                                           float, int)
    void _reconstruction_grayscale "reconstruction_grayscale"[dtype](dtype*, dtype*, dtype*, int, 
                                                                     int, int, int, MorphOp, int)
    void _bottom_hat "bottom_hat"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, int, 
                                         float, int)       
    void _top_hat "top_hat"[dtype](dtype*, dtype*, int, int, int, int, int*, int, int, int, float,
                                   int)
    void _top_hat_reconstruction "top_hat_reconstruction"[dtype](dtype*, dtype*, int, int, int, int, 
                                                                 int*, int, int, int, int)
    void _bottom_hat_reconstruction "bottom_hat_reconstruction"[dtype](dtype*, dtype*, int, int, 
                                                                       int, int, int*, int, int, 
                                                                       int, int)

@boundscheck(False)
@wraparound(False)
def erosion(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint, 
                      numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                      float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale erosion on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
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
       Chapter 9 (Morphological Image Processing), Section 9.8, on pages 674-679.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)
    _erosion_grayscale(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose, 
                       &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def dilation(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint, 
                       numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                       float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale dilation on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
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
       Chapter 9 (Morphological Image Processing), Section 9.8, on pages 674-679.
    """ 

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)
    _dilation_grayscale(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose, 
                        &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def closing(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint, 
                      numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                      float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale closing on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
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
       Chapter 9 (Morphological Image Processing), Section 9.8, pages 680-682.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)
    
    _closing_grayscale(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose,
                       &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def opening(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint, 
                      numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                      float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale openning on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
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
       Chapter 9 (Morphological Image Processing), Section 9.8, pages 680-682.
    """ 

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)
    
    _opening_grayscale(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose,
                       &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def geodesic_erosion(numpy.ndarray[numeric, ndim=3] image, 
                            numpy.ndarray[numeric, ndim=3] mask, 
                            numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                            float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale geodesic erosion on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param mask: 3D mask image that acts as a constraint, limiting the extent of erosion 
                     on the input image.    
    :type mask: numpy.ndarray[numeric, ndim=3]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
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
       Chapter 9 (Morphological Image Processing), Section 9.6, pages 667-668.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    
    _geodesic_erosion_grayscale(&image[0,0,0], &mask[0,0,0], &out[0,0,0], isize.x, 
                             isize.y, isize.z, verbose, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def geodesic_dilation(numpy.ndarray[numeric, ndim=3] image, 
                             numpy.ndarray[numeric, ndim=3] mask, 
                             numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                             float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale geodesic dilation on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param mask: 3D mask image that acts as a constraint, limiting the extent of dilation 
                     on the input image.    
    :type mask: numpy.ndarray[numeric, ndim=3]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
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
       Chapter 9 (Morphological Image Processing), Section 9.6, pages 667-668.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)

    _geodesic_dilation_grayscale(&image[0,0,0], &mask[0,0,0], &out[0,0,0], isize.x, 
                              isize.y, isize.z, verbose, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def reconstruction(numpy.ndarray[numeric, ndim=3] image, 
                          numpy.ndarray[numeric, ndim=3] mask, str method, 
                          numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                          int ngpus = -1):
    """
    Performs morphological reconstruction on a grayscale 3D image using erosion or dilation.

    :param image: Input 3D grayscale image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param mask: 3D mask image that acts as a constraint, limiting the extent of the method.
    :type mask: numpy.ndarray[numeric, ndim=3]
    :param method: Morphological method to perform. Must be either 'erosion' or 'dilation'.
    :type method: str
    :param out: Output 3D image. If None, a new array with the same shape as `image` is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param ngpus: Whether to execute on GPU or CPU. 
                  If ngpus = 0, CPU execution is selected. 
                  Otherwise, the function executes on GPU.
    :type ngpus: int, default -1
    :raises ValueError: If `method` is not 'erosion' or 'dilation'.
    :return: The reconstructed 3D grayscale image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.8, on pages 688-691.
    """ 

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)

    cdef MorphOp morph_op
    if method == "erosion":
        morph_op = EROSION
    elif method == "dilation":
        morph_op = DILATION
    else:
        raise ValueError("Invalid method. Must be 'erosion' or 'dilation'.")
    
    if out is None:
        out = numpy.empty_like(image)

    _reconstruction_grayscale(&image[0,0,0], &mask[0,0,0], &out[0,0,0], isize.x, 
                           isize.y, isize.z, verbose, morph_op, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def white_tophat(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint,
            numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
            float gpuMemory = 0.4, int ngpus = -1):
    """
    Applies the top-hat transform to a 3D image using a given footprint.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element used for morphological processing.
    :type footprint: int[:,:,:]
    :param out: Output 3D image. If None, a new array with the same shape as `image` is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param gpuMemory: Fraction of available GPU memory to use (between 0 and 1).
    :type gpuMemory: float, optional
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The top-hat transformed 3D image.
    :rtype: numpy.ndarray[numeric, ndim=3]

    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.8, pages 683-685.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _top_hat(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose, 
             &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def black_tophat(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint,
               numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
               float gpuMemory = 0.4, int ngpus = -1):
    """
    Applies the bottom-hat transform to a 3D image using a given footprint.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element used for morphological processing.
    :type footprint: int[:,:,:]
    :param out: Output 3D image. If None, a new array with the same shape as `image` is created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param gpuMemory: Fraction of available GPU memory to use (between 0 and 1).
    :type gpuMemory: float, optional
    :param ngpus: The number of GPUs to use. 
                  If ngpus < 1, all available GPUs are used.
                  If ngpus = 0, CPU execution is selected. 
                  If ngpus >= 1, the function uses up to min(ngpus, available GPUs).
    :type ngpus: int, default -1
    :return: The bottom-hat transformed 3D image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.8, pages 683-685.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _bottom_hat(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose, 
                &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)             
    return out

@boundscheck(False)
@wraparound(False)
def white_tophat_reconstruction(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint,
            numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, int ngpus = -1):
    """
    Performs a specialized top-hat transformation of a grayscale 3D image. This version applies 
    morphological reconstruction to preserve edge details, ensuring accurate segmentation of 
    porosity.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element used for morphological processing.
    :type footprint: int[:,:,:]
    :param out: Output 3D image. If None, a new array with the same shape as `image` is 
                       created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param ngpus: Whether to execute on GPU or CPU. 
                  If ngpus = 0, CPU execution is selected. 
                  Otherwise, the function executes on GPU.
    :type ngpus: int, default -1
    :return: The reconstructed 3D image after the top-hat transform.
    :rtype: numpy.ndarray[numeric, ndim=3]

    .. note::
       This implementation is inspired by the Interactive Top-Hat by Reconstruction module,
       which enhances segmentation by applying grayscale reconstruction techniques.
       Reference: `<https://www.thermofisher.com/software-em-3d-vis/xtra-library/xtras/interactive-top-hat-by-reconstruction>`_
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _top_hat_reconstruction(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose, 
             &footprint[0,0,0], ksize.x, ksize.y, ksize.z, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def black_tophat_reconstruction(numpy.ndarray[numeric, ndim=3] image, int[:,:,:] footprint,
                              numpy.ndarray[numeric, ndim=3] out = None, int verbose = 0, 
                              int ngpus = -1):
    """
    Performs a specialized bottom-hat transformation of a grayscale 3D image. This version applies 
    morphological reconstruction to preserve edge details, ensuring accurate segmentation of 
    porosity.

    :param image: Input 3D image.
    :type image: numpy.ndarray[numeric, ndim=3]
    :param footprint: Structuring element used for morphological processing.
    :type footprint: int[:,:,:]
    :param out: Output 3D image. If None, a new array with the same shape as `image` is 
                       created.
    :type out: numpy.ndarray[numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param ngpus: Whether to execute on GPU or CPU. 
                  If ngpus = 0, CPU execution is selected. 
                  Otherwise, the function executes on GPU.
    :type ngpus: int, default -1
    :return: The reconstructed 3D image after the bottom-hat transform.
    :rtype: numpy.ndarray[numeric, ndim=3]

    .. note::
       This implementation is inspired by the Interactive Top-Hat by Reconstruction module,
       which enhances segmentation by applying grayscale reconstruction techniques.
       Reference: `<https://www.thermofisher.com/software-em-3d-vis/xtra-library/xtras/interactive-top-hat-by-reconstruction>`_
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _bottom_hat_reconstruction(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, 
                               verbose, &footprint[0,0,0], ksize.x, ksize.y, ksize.z, ngpus)

    return out


####################################################################################################
#----------------------------------BINARY OPERATIONS------------------------------------------------
####################################################################################################

ctypedef fused binary_numeric:
    int
    unsigned int
    int16_t
    uint16_t

cdef extern from "../include/morphology/operations_binary.h":
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

@boundscheck(False)
@wraparound(False)
def binary_erosion(numpy.ndarray[binary_numeric, ndim=3] image, int[:,:,:] footprint, 
                   numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary erosion on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.2, on pages 638-643.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _erosion_binary(&image[0,0,0], &out[0,0,0],  isize.x, isize.y, isize.z, 
                    verbose, &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)    
def binary_dilation(numpy.ndarray[binary_numeric, ndim=3] image, int[:,:,:] footprint, 
                    numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                    float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary dilation on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.2, on pages 638-643.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _dilation_binary(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, 
                     verbose, &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def binary_closing(numpy.ndarray[binary_numeric, ndim=3] image, int[:,:,:] footprint, 
                   numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary closing on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.3, on pages 644-648.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _closing_binary(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, 
                    verbose, &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def binary_opening(numpy.ndarray[binary_numeric, ndim=3] image, int[:,:,:] footprint, 
                   numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary openning on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.3, on pages 644-648.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _opening_binary(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, 
                    verbose, &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def binary_smoothing(numpy.ndarray[binary_numeric, ndim=3] image, int[:,:,:] footprint, 
                   numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                   float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary smoothing on a 3D image. The smooth operation consists of a 
    sequence of openning and closing operations.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param footprint: Structuring element for erosion.
    :type footprint: int[:, :, :]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation is based on the morphological operations described in "Digital Image 
       Processing, 4th Edition" by R.C. Gonzalez and R.E. Woods, particularly in Chapter 9 
       (Morphological Image Processing), Section 9.8, on page 682. The grayscale smoothing 
       algorithm described was adapted for binary images.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    ksize = Size(footprint)

    _smooth_binary(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, verbose, 
                   &footprint[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def binary_geodesic_erosion(numpy.ndarray[binary_numeric, ndim=3] image, 
                            numpy.ndarray[binary_numeric, ndim=3] mask, 
                            numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                            float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary geodesic erosion on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param mask: 3D mask image that acts as a constraint, limiting the extent of erosion 
                     on the input image.    
    :type mask: numpy.ndarray[binary_numeric, ndim=3]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 667-668.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    
    _geodesic_erosion_binary(&image[0,0,0], &mask[0,0,0], &out[0,0,0], isize.x, 
                             isize.y, isize.z, verbose, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def binary_geodesic_dilation(numpy.ndarray[binary_numeric, ndim=3] image, 
                             numpy.ndarray[binary_numeric, ndim=3] mask, 
                             numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                             float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs binary geodesic dilation on a 3D image.

    :param image: Input 3D image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param mask: 3D mask image that acts as a constraint, limiting the extent of dilation 
                     on the input image.    
    :type mask: numpy.ndarray[binary_numeric, ndim=3]
    :param out: Optional output array. If None, a new array is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 667-668.
    """

    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)

    _geodesic_dilation_binary(&image[0,0,0], &mask[0,0,0], &out[0,0,0], isize.x, 
                              isize.y, isize.z, verbose, gpuMemory, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def binary_reconstruction(numpy.ndarray[binary_numeric, ndim=3] seed, 
                          numpy.ndarray[binary_numeric, ndim=3] mask, str method, 
                          numpy.ndarray[binary_numeric, ndim=3] out = None, int verbose = 0, 
                          int ngpus = -1):
    """
    Performs morphological reconstruction on a binary 3D image using erosion or dilation.

    :param seed: Input 3D binary seed.
    :type seed: numpy.ndarray[binary_numeric, ndim=3]
    :param mask: 3D mask image that acts as a constraint, limiting the extent of the method.
    :type mask: numpy.ndarray[binary_numeric, ndim=3]
    :param method: Morphological method to perform. Must be either 'erosion' or 'dilation'.
    :type method: str
    :param out: Output 3D image. If None, a new array with the same shape as `image` is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
    :param verbose: Verbosity level for debugging output.
    :type verbose: int, optional
    :param ngpus: Whether to execute on GPU or CPU. 
                  If ngpus = 0, CPU execution is selected. 
                  Otherwise, the function executes on GPU.
    :type ngpus: int, default -1
    :raises ValueError: If `method` is not 'erosion' or 'dilation'.
    :return: The reconstructed 3D binary image.
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 668-669.
    """ 

    if out is None:
        out = numpy.empty_like(seed)

    isize = Size(seed)

    cdef MorphOp morph_op
    if method == "erosion":
        morph_op = EROSION
    elif method == "dilation":
        morph_op = DILATION
    else:
        raise ValueError("Invalid method. Must be 'erosion' or 'dilation'.")
    
    _reconstruction_binary(&seed[0,0,0], &mask[0,0,0], &out[0,0,0], isize.x, 
                           isize.y, isize.z, verbose, morph_op, ngpus)

    return out

@boundscheck(False)
@wraparound(False)
def fill_holes(numpy.ndarray[binary_numeric, ndim=3] image, 
               numpy.ndarray[binary_numeric, ndim=3] out = None,  int padding = 50, int verbose = 0, 
               float gpuMemory = 0.4, int ngpus = -1):
    """
    Fills holes in a binary 3D image.

    :param image: Input 3D binary image.
    :type image: numpy.ndarray[binary_numeric, ndim=3]
    :param out: Output 3D image. If None, a new array with the same shape as `image` is created.
    :type out: numpy.ndarray[binary_numeric, ndim=3], optional
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
    :rtype: numpy.ndarray[binary_numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.6, on pages 671-672.
    """
        
    if out is None:
        out = numpy.empty_like(image)

    isize = Size(image)
    
    _fill_holes(&image[0,0,0], &out[0,0,0], isize.x, isize.y, isize.z, padding, verbose, 
                gpuMemory, ngpus)

    return out
