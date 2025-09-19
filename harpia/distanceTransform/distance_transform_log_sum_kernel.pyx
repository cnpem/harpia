cimport cython
from cython.parallel import prange
import numpy 
cimport numpy
from harpia.common import Size


ctypedef fused numeric:
    int
    unsigned int

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_fftfreq(int n, double d = 1.0):
    """
    This function computes the frequencies for a 1D FFT.
    n: size of the input
    d: sample spacing
    """
    cdef numpy.ndarray[numpy.float64_t, ndim=1] result = numpy.empty(n, dtype=numpy.float64)
    cdef int i

    # Compute half of the frequency bins (0 to n//2)
    for i in prange(n // 2 + 1, nogil=True):
        result[i] = i / (n * d)

    # Compute the negative half (-(n//2-1) to -1)
    for i in prange(n // 2 + 1, n, nogil=True):
        result[i] = (i - n) / (n * d)

    return result


#parallel element-wise multiplication

def distance_transform_edt(numpy.ndarray[numeric, ndim=3] hostImage,
                           numpy.ndarray[numpy.float32_t, ndim=3] hostOutput = None,
                           float lmbd = 1e-3, float thresh = 1,
                           int verbose = 0):
    """
    Distance transform using a Gaussian-like kernel defined directly in frequency space.

    Parameters:
        hostImage (ndarray): Input 3D binary image.
        hostOutput (ndarray, optional): Output array (float32) to store result. Auto-created if None.
        lmbd (float): Lambda parameter for log-sum approximation.
        thresh (float): Threshold for frequency cutoff.
        verbose (int): Verbosity level.

    Returns:
        ndarray: Distance transform result.
    """

    # Get image dimensions
    isize = Size(hostImage)

    if hostOutput is None:
        hostOutput = numpy.empty((isize.z, isize.y, isize.x), dtype=numpy.float32)

    # FFT of the inverted binary image (constant padding to avoid reflection issues)
    imageFFT = numpy.fft.fftn(numpy.logical_not(hostImage).astype(numpy.float32),
                              s=(isize.z, isize.y, isize.x)).astype(numpy.complex64)
    if verbose:
        print("Image FFT completed")

    # Frequency grid
    fz = parallel_fftfreq(isize.z)[:, numpy.newaxis, numpy.newaxis]
    fy = parallel_fftfreq(isize.y)[numpy.newaxis, :, numpy.newaxis]
    fx = parallel_fftfreq(isize.x)[numpy.newaxis, numpy.newaxis, :]

    d = fx**2 + fy**2 + fz**2
    d = numpy.where(d > thresh, numpy.inf, d)

    # Gaussian-like filter in frequency space
    gaussian_filter_freq = numpy.exp(-(d / lmbd))
    if verbose:
        print("Frequency domain Gaussian filter created")

    # Convolution in frequency space
    convolvedImageFFT = imageFFT * gaussian_filter_freq
    if verbose:
        print("Convolution in frequency space completed")

    # Inverse FFT
    convolvedImage = numpy.fft.irfftn(convolvedImageFFT,
                                      s=(isize.z, isize.y, isize.x)).astype(numpy.float32)
    if verbose:
        print("Inverse FFT completed")

    # Clip to avoid log of zero
    convolvedImage = numpy.clip(convolvedImage, 1e-32, None)
    if verbose:
        print("Clipping completed")

    # Distance transform
    hostOutput[:] = (-lmbd * numpy.log(convolvedImage)).astype(numpy.float32)
    if verbose:
        print("Distance transform completed")

    return hostOutput
