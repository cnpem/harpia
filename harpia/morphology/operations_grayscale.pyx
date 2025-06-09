cimport numpy
import numpy

from libcpp cimport int
from cython cimport boundscheck, wraparound, parallel

from harpia.common import Size

ctypedef fused numeric:
    float
    int
    unsigned int

cdef extern from "../../include/morphology/morphology.h":
    ctypedef enum MorphOp:
        EROSION
        DILATION

cdef extern from "../../include/morphology/operations_grayscale.h":
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
def erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                      numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                      float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale erosion on a 3D image.

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
       Chapter 9 (Morphological Image Processing), Section 9.8, on pages 674-679.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)
    _erosion_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                       &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                       numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                       float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale dilation on a 3D image.

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
       Chapter 9 (Morphological Image Processing), Section 9.8, on pages 674-679.
    """ 

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)
    _dilation_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                        &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def closing_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                      numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                      float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale closing on a 3D image.

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
       Chapter 9 (Morphological Image Processing), Section 9.8, pages 680-682.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)
    
    _closing_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose,
                       &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def opening_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel, 
                      numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                      float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale openning on a 3D image.

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
       Chapter 9 (Morphological Image Processing), Section 9.8, pages 680-682.
    """ 

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)
    
    _opening_grayscale(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose,
                       &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def geodesic_erosion_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, 
                            numpy.ndarray[numeric, ndim=3] hostMask, 
                            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                            float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale geodesic erosion on a 3D image.

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
       Chapter 9 (Morphological Image Processing), Section 9.6, pages 667-668.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    
    _geodesic_erosion_grayscale(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                             isize.y, isize.z, verbose, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def geodesic_dilation_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, 
                             numpy.ndarray[numeric, ndim=3] hostMask, 
                             numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                             float gpuMemory = 0.4, int ngpus = -1):
    """
    Performs grayscale geodesic dilation on a 3D image.

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
       Chapter 9 (Morphological Image Processing), Section 9.6, pages 667-668.
    """

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)

    _geodesic_dilation_grayscale(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                              isize.y, isize.z, verbose, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def reconstruction_grayscale(numpy.ndarray[numeric, ndim=3] hostImage, 
                          numpy.ndarray[numeric, ndim=3] hostMask, str operation, 
                          numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                          int ngpus = -1):
    """
    Performs morphological reconstruction on a grayscale 3D image using erosion or dilation.

    :param hostImage: Input 3D grayscale image.
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
    :return: The reconstructed 3D grayscale image.
    :rtype: numpy.ndarray[numeric, ndim=3]
    
    .. note::
       This implementation follows the morphological transformation principles described in:
       R.C. Gonzalez, R.E. Woods, "Digital Image Processing," 4th Edition, Pearson, 2018.
       Chapter 9 (Morphological Image Processing), Section 9.8, on pages 688-691.
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
    
    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    _reconstruction_grayscale(&hostImage[0,0,0], &hostMask[0,0,0], &hostOutput[0,0,0], isize.x, 
                           isize.y, isize.z, verbose, morph_op, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def top_hat(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
            float gpuMemory = 0.4, int ngpus = -1):
    """
    Applies the top-hat transform to a 3D image using a given kernel.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element used for morphological processing.
    :type kernel: int[:,:,:]
    :param hostOutput: Output 3D image. If None, a new array with the same shape as `hostImage` is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
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

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _top_hat(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
             &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def bottom_hat(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
               numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
               float gpuMemory = 0.4, int ngpus = -1):
    """
    Applies the bottom-hat transform to a 3D image using a given kernel.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element used for morphological processing.
    :type kernel: int[:,:,:]
    :param hostOutput: Output 3D image. If None, a new array with the same shape as `hostImage` is created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
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

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _bottom_hat(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
                &kernel[0,0,0], ksize.x, ksize.y, ksize.z, gpuMemory, ngpus)             
    return hostOutput

@boundscheck(False)
@wraparound(False)
def top_hat_reconstruction(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
            numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, int ngpus = -1):
    """
    Performs a specialized top-hat transformation of a grayscale 3D image. This version applies 
    morphological reconstruction to preserve edge details, ensuring accurate segmentation of 
    porosity.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element used for morphological processing.
    :type kernel: int[:,:,:]
    :param hostOutput: Output 3D image. If None, a new array with the same shape as `hostImage` is 
                       created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
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

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _top_hat_reconstruction(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, verbose, 
             &kernel[0,0,0], ksize.x, ksize.y, ksize.z, ngpus)

    return hostOutput

@boundscheck(False)
@wraparound(False)
def bottom_hat_reconstruction(numpy.ndarray[numeric, ndim=3] hostImage, int[:,:,:] kernel,
                              numpy.ndarray[numeric, ndim=3] hostOutput = None, int verbose = 0, 
                              int ngpus = -1):
    """
    Performs a specialized bottom-hat transformation of a grayscale 3D image. This version applies 
    morphological reconstruction to preserve edge details, ensuring accurate segmentation of 
    porosity.

    :param hostImage: Input 3D image.
    :type hostImage: numpy.ndarray[numeric, ndim=3]
    :param kernel: Structuring element used for morphological processing.
    :type kernel: int[:,:,:]
    :param hostOutput: Output 3D image. If None, a new array with the same shape as `hostImage` is 
                       created.
    :type hostOutput: numpy.ndarray[numeric, ndim=3], optional
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

    if hostOutput is None:
        hostOutput = numpy.empty_like(hostImage)

    isize = Size(hostImage)
    ksize = Size(kernel)

    _bottom_hat_reconstruction(&hostImage[0,0,0], &hostOutput[0,0,0], isize.x, isize.y, isize.z, 
                               verbose, &kernel[0,0,0], ksize.x, ksize.y, ksize.z, ngpus)

    return hostOutput