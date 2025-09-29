cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool
from harpia.common import Size

cdef extern from "../include/morphology/morph_snakes_2d.h":
    void morph_geodesic_active_contour(float* hostImage, bool* initLs, const int iterations, const float balloonForce, const float threshold, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose)

    void morph_chan_vese(float* hostImage, bool* initLs, const int iterations, const float lambda1, const float lambda2, const int smoothing, bool* hostOutput,
                         const int xsize, const int ysize,
                         const int flag_verbose)
    
def morphological_geodesic_active_contour(np.ndarray[np.float32_t, ndim=2] gimage, int num_iter, np.ndarray[bool, ndim=2] init_level_set, int smoothing = 1, float threshold = 0.4, float balloon = 1, int flag_verbose=0):
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
    num_iter : int
        Number of iterations to run
    init_level_set : (ysize, xsize) bool array
        Initial level set. 
    smoothing : int, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
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
    flag_verbose : bool, optional
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
    #apply padding
    gimage = np.pad(gimage, pad_width=1, mode="edge")
    init_level_set = np.pad(init_level_set, pad_width=1, mode="edge")
    # Ensure input arrays are C-contiguous
    gimage = np.ascontiguousarray(gimage, dtype=np.float32)
    init_level_set = np.ascontiguousarray(init_level_set, dtype=np.bool_)

    # Define variables
    isize = Size(gimage)

    # Create the output array
    cdef np.ndarray[bool, ndim=2] hostOutput = np.zeros((isize.y, isize.x), dtype=np.bool_)

    # Call the external C function
    morph_geodesic_active_contour(
        &gimage[0, 0], &init_level_set[0, 0], num_iter, balloon, threshold, smoothing,
        &hostOutput[0, 0], isize.x, isize.y, flag_verbose
    )

    return hostOutput[1:-1, 1:-1]

def morphological_chan_vese(np.ndarray[np.float32_t, ndim=2] image, int num_iter, np.ndarray[bool, ndim=2] init_level_set, int smoothing = 1, float lambda1 = 1, float lambda2 = 1, int flag_verbose = 0):
    """Morphological Active Contours without Edges (MorphACWE)

    Active contours without edges implemented with morphological operators. It
    can be used to segment objects in images and volumes without well defined
    borders. It is required that the inside of the object looks different on
    average than the outside (i.e., the inner area of the object should be
    darker or lighter than the outer area on average).

    Parameters
    ----------
    image : (ysize, xsize) float array
        Grayscale image or volume to be segmented.
    num_iter : int
        Number of iterations to run
    init_level_set : (ysize, xsize) bool array
        Initial level set. 
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
    flag_verbose : bool, optional
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
           Transactions on Pattern Analysis and Machine Intelligence (PANI),
           2014, DOI 10.1109/TPAMI.2013.106
    """

    #apply padding
    image = np.pad(image, pad_width=1, mode="edge")
    init_level_set = np.pad(init_level_set, pad_width=1, mode="edge")

    # Ensure input arrays are C-contiguous
    image = np.ascontiguousarray(image, dtype=np.float32)
    init_level_set = np.ascontiguousarray(init_level_set, dtype=np.bool_)

    # Get image size
    isize = Size(image)

    # Create the output array
    cdef np.ndarray[bool, ndim=2] hostOutput = np.zeros((isize.y, isize.x), dtype=np.bool_)

    # Call the external C function
    morph_chan_vese(
        &image[0, 0], &init_level_set[0, 0], num_iter, lambda1, lambda2, smoothing,
        &hostOutput[0, 0], isize.x, isize.y, flag_verbose
    )

    return hostOutput[1:-1, 1:-1]


cdef extern from "../include/watershed/marker_based_watershed.h":
    void meyers_watershed_2d(int* hostImage, int* markers, int background, int xsize, int ysize)
    void meyers_watershed_3d(int* hostImage, int* markers, int background, int xsize, int ysize, int zsize)


def watershed_meyers_2d(np.ndarray[int, ndim=2] hostImage,
                        np.ndarray[int, ndim=2] markers,
                        int background):
    """
    Apply Meyer's watershed segmentation in 2D using GPU-based chunked processing.

    Parameters:
        hostImage (ndarray[int, ndim=2]): Input 2D image (usually gradient or intensity).
        markers  (ndarray[int, ndim=2]): Marker image (integer labels).
        background (int): Background label.
        hostOutput (ndarray[int, ndim=2], optional): Preallocated output label array. Auto-created if None.
        verbose (int): Verbosity level.
        gpuMemory (float): Fraction of GPU memory to use.
        ngpus (int): Number of GPUs (-1 = all available).

    Returns:
        ndarray[int, ndim=2]: Segmentation label image.
    """

    isize = Size(hostImage)

    meyers_watershed_2d(&hostImage[0,0],
                             &markers[0,0],
                             background,
                             isize.y, isize.x)

    return markers


def watershed_meyers_3d(np.ndarray[int, ndim=3] hostImage,
                        np.ndarray[int, ndim=3] markers,
                        int background):
    """
    Apply Meyer's watershed segmentation in 3D using GPU-based chunked processing.

    Parameters:
        hostImage (ndarray[int, ndim=3]): Input 3D image.
        markers  (ndarray[int, ndim=3]): Marker image (integer labels).
        background (int): Background label.
        hostOutput (ndarray[int, ndim=3], optional): Preallocated output label volume. Auto-created if None.
        verbose (int): Verbosity level.
        gpuMemory (float): Fraction of GPU memory to use.
        ngpus (int): Number of GPUs (-1 = all available).

    Returns:
        ndarray[int, ndim=3]: Segmentation label volume.
    """

    isize = Size(hostImage)

    meyers_watershed_3d(&hostImage[0,0,0],
                             &markers[0,0,0],
                             background,
                             isize.y, isize.x, isize.z)

    return markers