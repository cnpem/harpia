cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool
from harpia.common import Size

cdef extern from "../../include/morphology/morph_snakes_2d.h":
    void morph_geodesic_active_contour(float* hostImage, bool* initLs, const int iterations, const float balloonForce, const float threshold, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose)

    void morph_chan_vese(float* hostImage, bool* initLs, const int iterations, const float lambda1, const float lambda2, const int smoothing, bool* hostOutput,
                         const int xsize, const int ysize,
                         const int flag_verbose)

def morph_2D_geodesic_active_contour(np.ndarray[np.float32_t, ndim=2] hostImage, np.ndarray[bool, ndim=2] initLs, int iterations, float threshold, float balloonForce, int smoothing, int flag_verbose=0):
    """Morphological Geodesic Active Contours (MorphGAC).

    Geodesic active contours implemented with morphological operators. It can
    be used to segment objects with visible but noisy, cluttered, broken
    borders.

    Parameters
    ----------
    gimage : (ysize, xsize) float array
        Preprocessed image or volume to be segmented. This is very rarely the
        original image. Instead, this is usually a preprocessed version of the
        original image that enhances and highlights the borders (or other
        structures) of the object to segment.
        `morphological_geodesic_active_contour` will try to stop the contour
        evolution in areas where `gimage` is small. See scikit-image
        `morphsnakes.inverse_gaussian_gradient` as an example function to
        perform this preprocessing. Note that the quality of
        `morphological_geodesic_active_contour` might greatly depend on this
        preprocessing.
    initLs :(ysize, xsize) bool array
        Initial level set. 
    iterations : int
        Number of iterations to run
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
        Effectively in the code, it only matter if it's positive, negative or zero.
    smoothing : int, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    flag_verbose: bool, optional
        If set to a non-zero value, the function will print
        the grid and block dimensions used for kernel execution to the console. This
        is useful for debugging and performance analysis to understand how the computation
        is distributed across CUDA threads.

    Returns
    -------
    out : (ysize, xsize) bool array
        Final segmentation (i.e., the final level set)

    Notes
    -----

    This is a version of the Geodesic Active Contours (GAC) algorithm that uses
    morphological operators instead of solving partial differential equations
    (PDEs) for the evolution of the contour. The set of morphological operators
    used in this algorithm are proved to be infinitesimally equivalent to the
    GAC PDEs (see [1]_). However, morphological operators are do not suffer
    from the numerical stability issues typically found in PDEs (e.g., it is
    not necessary to find the right time step for the evolution), and are
    computationally faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """
    # Ensure input arrays are C-contiguous
    hostImage = np.ascontiguousarray(hostImage, dtype=np.float32)
    initLs = np.ascontiguousarray(initLs, dtype=np.bool_)

    # Define variables
    isize = Size(hostImage)

    # Create the output array
    cdef np.ndarray[bool, ndim=2] hostOutput = np.zeros((isize.y, isize.x), dtype=np.bool_)

    # Call the external C function
    morph_geodesic_active_contour(
        &hostImage[0, 0], &initLs[0, 0], iterations, balloonForce, threshold, smoothing,
        &hostOutput[0, 0], isize.x, isize.y, flag_verbose
    )

    return hostOutput

def morph_2D_chan_vese(np.ndarray[np.float32_t, ndim=2] hostImage, np.ndarray[bool, ndim=2] initLs, int iterations, float lambda1, float lambda2, int smoothing, int flag_verbose=0):
    """Morphological Active Contours without Edges (MorphACWE)

    Active contours without edges implemented with morphological operators. It
    can be used to segment objects in images and volumes without well defined
    borders. It is required that the inside of the object looks different on
    average than the outside (i.e., the inner area of the object should be
    darker or lighter than the outer area on average).

    Parameters
    ----------
    hostImage : (ysize, xsize) float array
        Grayscale image or volume to be segmented.
    initLs :(ysize, xsize) bool array
        Initial level set. 
    iterations : int
        Number of iterations to run
    smoothing : int, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    lambda1 : float, optional
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
    lambda2 : float, optional
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    flag_verbose: bool, optional
        If set to a non-zero value, the function will print
        the grid and block dimensions used for kernel execution to the console. This
        is useful for debugging and performance analysis to understand how the computation
        is distributed across CUDA threads.

    Returns
    -------
    out : (ysize, xsize) bool array
        Final segmentation (i.e., the final level set)

    Notes
    -----

    This is a version of the Chan-Vese algorithm that uses morphological
    operators instead of solving a partial differential equation (PDE) for the
    evolution of the contour. The set of morphological operators used in this
    algorithm are proved to be infinitesimally equivalent to the Chan-Vese PDE
    (see [1]_). However, morphological operators are do not suffer from the
    numerical stability issues typically found in PDEs (it is not necessary to
    find the right time step for the evolution), and are computationally
    faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """
    # Ensure input arrays are C-contiguous
    hostImage = np.ascontiguousarray(hostImage, dtype=np.float32)
    initLs = np.ascontiguousarray(initLs, dtype=np.bool_)

    # Get image size
    isize = Size(hostImage)

    # Create the output array
    cdef np.ndarray[bool, ndim=2] hostOutput = np.zeros((isize.y, isize.x), dtype=np.bool_)

    # Call the external C function
    morph_chan_vese(
        &hostImage[0, 0], &initLs[0, 0], iterations, lambda1, lambda2, smoothing,
        &hostOutput[0, 0], isize.x, isize.y, flag_verbose
    )

    return hostOutput
