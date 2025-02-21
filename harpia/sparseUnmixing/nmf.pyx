import numpy as np
cimport numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from cython.parallel import prange
from scipy.ndimage import uniform_filter
from cython cimport boundscheck, wraparound, parallel
from time import perf_counter
from libc.math cimport sqrt

ctypedef fused numeric:
    int
    unsigned int
    float

@boundscheck(False)
@wraparound(False)
def solver_nmf(np.ndarray[np.float32_t, ndim=2] X,
               np.ndarray[np.float32_t, ndim=2] D,
               np.ndarray[np.float32_t, ndim=2] V,
               float beta, int iterations):

    cdef int P = X.shape[1]  # Number of pixels
    cdef int B = X.shape[0]  # Number of features/bands
    cdef int M = D.shape[1]  # Number of endmembers

    #to enforce sum-to-one constraint
    cdef np.ndarray[np.float32_t, ndim=2] Xbar = np.vstack([X, np.ones((1, P), dtype=np.float32)])
    cdef np.ndarray[np.float32_t, ndim=2] Abar = np.vstack([D, np.ones((1, M), dtype=np.float32)])#np.empty((B+1, M), dtype=np.float32)

    #pre-allocation
    cdef np.ndarray[np.float32_t, ndim=2] Vnumerator = np.empty((M, P), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Vdenominator = np.empty((M, P), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] AbarTAbar = np.zeros((M, M), dtype=np.float32)

    cdef int i, j, k
    cdef float sqrtV

    for i in prange(M,nogil=True):
        for j in range(M):
            AbarTAbar[i, j] = 0.0
            for k in range(B + 1):
                AbarTAbar[i, j] += Abar[k, i] * Abar[k, j]

    # Iterative update loop
    for _ in range(iterations):

        # Compute Vnumerator = Abar^T * Xbar
        for i in range(M):
            for j in range(P):
                Vnumerator[i, j] = 0.0
                for k in range(B + 1):
                    Vnumerator[i, j] += Abar[k, i] * Xbar[k, j]

        # Compute Vdenominator = (AbarTAbar @ V) + (beta / 2) / sqrt(V)
        for i in prange(M, nogil=True):
            for j in range(P):
                Vdenominator[i, j] = 0.0
                for k in range(M):
                    Vdenominator[i, j] += AbarTAbar[i, k] * V[k, j]

                # Regularization term
                sqrtV = max(sqrt(V[i, j]), 1e-6)  # Prevent division by zero
                Vdenominator[i, j] += (beta / 2) / sqrtV

        # Update V using multiplicative update rule
        for i in prange(M, nogil=True):
            for j in range(P):
                if Vdenominator[i, j] != 0:
                    V[i, j] *= Vnumerator[i, j] / (Vdenominator[i, j]+1e-16)

        # Ensure non-negativity
        for i in prange(M, nogil=True):
            for j in range(P):
                if V[i, j] < 0:
                    V[i, j] = 0