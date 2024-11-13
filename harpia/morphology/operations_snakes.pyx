cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool

cdef extern from "../../include/morphology/morph_snakes_2d.h":
    void morph_geodesic_active_contour(float* hostImage, bool* initLs, const int iterations, const float balloonForce, const float threshold, const int smoothing, bool* hostOutput,
                        const int xsize, const int ysize,
                        const int flag_verbose)

    void morph_chan_vese(float* hostImage, bool* initLs, const int iterations, const float lambda1, const float lambda2, const int smoothing, bool* hostOutput,
                         const int xsize, const int ysize,
                         const int flag_verbose)

def morph_2D_geodesic_active_contour(np.ndarray[np.float32_t, ndim=2] hostImage, np.ndarray[bool, ndim=2] initLs, int iterations, float threshold, float balloonForce, int smoothing, int flag_verbose=0):
    # Ensure input arrays are C-contiguous
    hostImage = np.ascontiguousarray(hostImage, dtype=np.float32)
    initLs = np.ascontiguousarray(initLs, dtype=np.bool_)

    # Define variables
    cdef int xsize = hostImage.shape[1]
    cdef int ysize = hostImage.shape[0]

    # Create the output array
    cdef np.ndarray[bool, ndim=2] hostOutput = np.zeros((xsize, ysize), dtype=np.bool_)

    # Call the external C function
    morph_geodesic_active_contour(
        &hostImage[0, 0], &initLs[0, 0], iterations, balloonForce, threshold, smoothing,
        &hostOutput[0, 0], xsize, ysize, flag_verbose
    )

    return hostOutput

def morph_2D_chan_vese(np.ndarray[np.float32_t, ndim=2] hostImage, np.ndarray[bool, ndim=2] initLs, int iterations, float lambda1, float lambda2, int smoothing, int flag_verbose=0):
    # Ensure input arrays are C-contiguous
    hostImage = np.ascontiguousarray(hostImage, dtype=np.float32)
    initLs = np.ascontiguousarray(initLs, dtype=np.bool_)

    # Define variables
    cdef int xsize = hostImage.shape[1]
    cdef int ysize = hostImage.shape[0]

    # Create the output array
    cdef np.ndarray[bool, ndim=2] hostOutput = np.zeros((xsize, ysize), dtype=np.bool_)

    # Call the external C function
    morph_chan_vese(
        &hostImage[0, 0], &initLs[0, 0], iterations, lambda1, lambda2, smoothing,
        &hostOutput[0, 0], xsize, ysize, flag_verbose
    )

    return hostOutput
