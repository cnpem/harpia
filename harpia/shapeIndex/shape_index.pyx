cimport cython
cimport numpy
import numpy

from libcpp cimport bool
from harpia.common import Size
from cython.parallel import prange
from cython cimport boundscheck, wraparound, parallel
from libc.math cimport sqrtf
from libc.math cimport atanf, M_PI

#Define the fused type for numeric types : float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int
#---------------------------------------------------------------------------------------------------
cdef extern from "../../include/shapeIndex/shape_index.h":
    void gradient[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, int zsize, int axis, float step)

def grad(numpy.ndarray[numeric, ndim=3] hostImage,
         numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
         int type3d=1,
         int verbose = 0, float gpuMemory = 0.4, int ngpus = -1,           
         int axis = 0, int step = 1):
    
    """
    Compute the gradient of a 3D image along a specified axis using CUDA.

    Parameters
    ----------
    hostImage : numpy.ndarray
        Input 3D image of numeric types (float, int, unsigned int), shape (z, y, x).
    hostOutput : numpy.ndarray, optional
        Pre-allocated output array (float32), same shape as hostImage.
        If None, a new array is allocated.
    type3d : int, optional
        Specifies whether to treat the input as 2D or 3D (currently unused).
    verbose : int, optional
        Verbosity level.
    gpuMemory : float, optional
        Fraction of GPU memory to allocate (reserved for future use).
    ngpus : int, optional
        Number of GPUs to use (-1 for automatic detection).
    axis : int, optional
        Axis along which to compute the gradient, y or x.
    step : int, optional
        Step size for finite differences.

    Returns
    -------
    numpy.ndarray
        Gradient of the input image along the specified axis, shape (z, y, x), dtype float32.
    """

    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    gradient(&hostImage[0, 0, 0],
                          &hostOutput[0, 0, 0],
                          isize.y, isize.x, isize.z, axis, step)
    
    return hostOutput



def hessian(numpy.ndarray[numeric, ndim=3] hostImage,
            numpy.ndarray[numpy.float32_t, ndim=3] hostOutput=None,
            int type3d=1,
            int verbose=0, float gpuMemory=0.4, int ngpus=-1,
            int axis=0, int step=1):

    """
    Compute the 2D Hessian components of a 3D image slice-wise.

    The Hessian matrix components returned are:
    - d²f/dx²
    - d²f/dy²
    - d²f/dxdy (mixed partial derivative)

    Parameters
    ----------
    hostImage : numpy.ndarray
        Input 3D image array.
    hostOutput : numpy.ndarray, optional
        Not currently used; reserved for output.
    type3d : int, optional
        Specifies whether to treat the input as 2D or 3D.
    verbose : int, optional
        Verbosity level.
    gpuMemory : float, optional
        Fraction of GPU memory to allocate.
    ngpus : int, optional
        Number of GPUs to use.
    axis : int, optional
        Axis parameter (not used here).
    step : int, optional
        Step size for finite differences.

    Returns
    -------
    tuple of numpy.ndarray
        Tuple with three 3D float32 arrays corresponding to (d²f/dx², d²f/dy², d²f/dxdy).
    """

    dfdx = grad(hostImage, step=step, axis=0,
                gpuMemory=gpuMemory, verbose=verbose, ngpus=ngpus)
    dfdxdy = grad(dfdx, step=step, axis=1,
                  gpuMemory=gpuMemory, verbose=verbose, ngpus=ngpus)
    dfdxdx = grad(dfdx, step=step, axis=0,
                  gpuMemory=gpuMemory, verbose=verbose, ngpus=ngpus)
    del dfdx


    dfdy = grad(hostImage, step=step, axis=1,
                gpuMemory=gpuMemory, verbose=verbose, ngpus=ngpus)
    dfdydy = grad(dfdy, step=step, axis=1,
                  gpuMemory=gpuMemory, verbose=verbose, ngpus=ngpus)
    del dfdy

    H = (dfdxdx, dfdydy, dfdxdy)

    return H

@boundscheck(False)
@wraparound(False)
def hessian_eigenvalues(numpy.ndarray[numeric, ndim=3] hostImage,
                        int step=1, int verbose=0,
                        float gpuMemory=0.4, int ngpus=-1):
    """
    Compute the two eigenvalues of the 2x2 Hessian matrix at each pixel for each slice in a 3D image.

    Parameters
    ----------
    hostImage : numpy.ndarray
        Input 3D image of numeric type.
    step : int, optional
        Finite difference step size.
    verbose : int, optional
        Verbosity flag.
    gpuMemory : float, optional
        Fraction of GPU memory to use.
    ngpus : int, optional
        Number of GPUs to use.

    Returns
    -------
    numpy.ndarray
        A 4D array of shape (z, y, x, 2), where the last dimension holds the eigenvalues (lambda1, lambda2).
    """
    cdef int z, y, x, i, j, k
    cdef float a, d, b, trace, delta, sqrt_delta, lambda1, lambda2

    cdef tuple H = hessian(hostImage, step=step, verbose=verbose, gpuMemory=gpuMemory, ngpus=ngpus)
    cdef numpy.ndarray[numpy.float32_t, ndim=3] dfdxdx = H[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=3] dfdydy = H[1]
    cdef numpy.ndarray[numpy.float32_t, ndim=3] dfdxdy = H[2]

    z = dfdxdx.shape[0]
    y = dfdxdx.shape[1]
    x = dfdxdx.shape[2]

    cdef numpy.ndarray[numpy.float32_t, ndim=4] eigen = numpy.empty((z, y, x, 2), dtype=numpy.float32)

    for i in prange(z, nogil=True):
        for j in range(y):
            for k in range(x):
                a = dfdxdx[i, j, k]
                d = dfdydy[i, j, k]
                b = dfdxdy[i, j, k]

                trace = a + d
                delta = (a - d) * (a - d) + 4 * b * b
                sqrt_delta = sqrtf(delta)

                lambda1 = (trace + sqrt_delta) * 0.5
                lambda2 = (trace - sqrt_delta) * 0.5

                eigen[i, j, k, 0] = lambda1
                eigen[i, j, k, 1] = lambda2

    return eigen

@boundscheck(False)
@wraparound(False)
def shape_index(numpy.ndarray[numeric, ndim=3] hostImage,
                numpy.ndarray[numpy.float32_t, ndim=4] eigen = None,
                int step=1, int verbose=0,
                float gpuMemory=0.4, int ngpus=-1):
    """
    Compute the shape index scalar field from the eigenvalues of the Hessian matrix of a 3D image.

    The shape index is defined as:
    S = (2 / pi) * arctan((lambda2 + lambda1) / (lambda2 - lambda1))
    where lambda1 and lambda2 are the Hessian eigenvalues at each pixel.

    Parameters
    ----------
    hostImage : numpy.ndarray
        Input 3D image array.
    step : int, optional
        Finite difference step size.
    verbose : int, optional
        Verbosity flag.
    gpuMemory : float, optional
        Fraction of GPU memory to use.
    ngpus : int, optional
        Number of GPUs to use.

    Returns
    -------
    numpy.ndarray
        3D array of shape (z, y, x) with the computed shape index values.
    """

    cdef int z, y, x, i, j, k
    cdef float l1, l2, num, den, sidx

    if eigen is None:
        eigen = hessian_eigenvalues(hostImage,
                                    step=step,
                                    verbose=verbose,
                                    gpuMemory=gpuMemory,
                                    ngpus=ngpus)

    z, y, x = eigen.shape[0], eigen.shape[1], eigen.shape[2]
    cdef numpy.ndarray[numpy.float32_t, ndim=3] shape = numpy.empty((z, y, x), dtype=numpy.float32)

    for i in prange(z, nogil=True):
        for j in range(y):
            for k in range(x):
                l1 = eigen[i, j, k, 0]
                l2 = eigen[i, j, k, 1]

                if l1 < l2:
                    l1, l2 = l2, l1

                num = l2 + l1
                den = l2 - l1

                if den == 0.0:
                    sidx = 0.0
                else:
                    sidx = (2.0 / M_PI) * atanf(num / den)

                shape[i, j, k] = sidx

    return shape
