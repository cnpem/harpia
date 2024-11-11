cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

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
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int i

    # Compute half of the frequency bins (0 to n//2)
    for i in prange(n // 2 + 1, nogil=True):
        result[i] = i / (n * d)

    # Compute the negative half (-(n//2-1) to -1)
    for i in prange(n // 2 + 1, n, nogil=True):
        result[i] = (i - n) / (n * d)

    return result


#parallel element-wise multiplication

def distance_transform(np.ndarray[numeric, ndim=3] hostImage,
                       float lmbd, float thresh,
                       int xsize, int ysize, int zsize):
    """
    Distance transform using a Gaussian-like kernel defined directly in frequency space.
    """

    # FFT of the image (with constant padding to avoid reflection issues)
    imageFFT = np.fft.fftn(np.logical_not(hostImage).astype(np.float32), s=(xsize, ysize, zsize)).astype(np.complex64)

    print("Image FFT completed")

    # Create the frequency grid, ensuring symmetry across all axes
    fx = parallel_fftfreq(xsize)[:, np.newaxis, np.newaxis]
    fy = parallel_fftfreq(ysize)[np.newaxis, :, np.newaxis]
    fz = parallel_fftfreq(zsize)[np.newaxis, np.newaxis, :]

    d = (fx**2 + fy**2 + fz**2)

    d = np.where(d > thresh, np.inf, d)
    # Create the Gaussian kernel directly in the frequency domain
    gaussian_filter_freq = np.exp(- (d / lmbd))

    print("Frequency domain Gaussian filter created")

    # Apply the Gaussian filter in frequency space
    convolvedImageFFT = imageFFT * gaussian_filter_freq

    print("Convolution in frequency space completed")

    # Inverse FFT to get the final result
    convolvedImage = np.fft.irfftn(convolvedImageFFT, s=(xsize, ysize, zsize)).astype(np.float32)

    print("Inverse FFT completed")

    # Clip to avoid log of zero
    convolvedImage = np.clip(convolvedImage, 1e-32, None)
    print('Clipping completed')

    # Calculate the distance transform
    distanceImage = -lmbd * np.log(convolvedImage).astype(np.float32)
    print('Distance transform completed')
    
    return distanceImage
