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
               np.ndarray[np.float32_t, ndim=2] Z,
               np.ndarray[np.float32_t, ndim=2] D,
               np.ndarray[np.float32_t, ndim=2] V,
               float beta, float gamma, int iterations):

    cdef int P = X.shape[1]  # Number of pixels
    cdef int B = X.shape[0]  # Number of features/bands
    cdef int M = D.shape[1]  # Number of endmembers

    # Enforce sum-to-one constraint
    cdef np.ndarray[np.float32_t, ndim=2] Xbar = np.vstack([X, np.ones((1, P), dtype=np.float32)])
    cdef np.ndarray[np.float32_t, ndim=2] Abar = np.vstack([D, np.ones((1, M), dtype=np.float32)])

    # Pre-allocation
    print('allocation')
    cdef np.ndarray[np.float32_t, ndim=2] Vnumerator = np.empty((M, P), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Vdenominator = np.empty((M, P), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] AbarTAbar = np.zeros((M, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] VZtDiag = np.zeros((M, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] VZtDiagZ = np.zeros((M, P), dtype=np.float32)
    
    # Ensure Diag is correctly shaped as MxM
    cdef np.ndarray[np.float32_t, ndim=2] Diag = np.diag(np.sum(Z, axis=1)).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] ZtDiag = np.zeros((P, M), dtype=np.float32)
    
    cdef int i, j, k
    cdef float sqrtV

    for i in prange(M, nogil=True):
        for j in range(M):
            AbarTAbar[i, j] = 0.0
            for k in range(B + 1):
                AbarTAbar[i, j] += Abar[k, i] * Abar[k, j]

    # Compute ZtDiag = Z.T * Diag
    for i in prange(P, nogil=True):  # Iterate over P
        for j in range(M):
            ZtDiag[i, j] = Z[j, i] * Diag[j, j]  # Use Z[j, i] for transposed indexing

    print("Diag: ", Diag)
    print("ZtDiag: ", ZtDiag)

    # Iterative update loop
    for _ in range(iterations):
        # Compute V * (Z^T * Diag)
        for i in prange(M, nogil=True):
            for j in range(M):
                VZtDiag[i, j] = 0.0
                for k in range(P):
                    VZtDiag[i, j] += V[i, k] * ZtDiag[k, j]

        print('VZtDiag: ', VZtDiag)

        # Compute VZtDiagZ = VZtDiag * Z (size MxP)
        for i in prange(M, nogil=True):
            for j in range(P):
                VZtDiagZ[i, j] = 0.0
                for k in range(M):  # Fix: Iterate over M, not P
                    VZtDiagZ[i, j] += VZtDiag[i, k] * Z[k, j]

        print('VZtDiagZ: ', VZtDiagZ)

        # Compute Vnumerator = Abar^T * Xbar
        for i in range(M):
            for j in range(P):
                Vnumerator[i, j] = 0.0
                for k in range(B + 1):
                    Vnumerator[i, j] += Abar[k, i] * Xbar[k, j]

        # Add gamma * VZTDiagZ to Vnumerator
        for i in prange(M, nogil=True):
            for j in range(P):
                Vnumerator[i, j] += gamma * VZtDiagZ[i, j]

        # Compute Vdenominator
        for i in prange(M, nogil=True):
            for j in range(P):
                Vdenominator[i, j] = 0.0
                for k in range(M):
                    Vdenominator[i, j] += AbarTAbar[i, k] * V[k, j]

                # Regularization term
                sqrtV = max(sqrt(V[i, j]), 1e-6)  # Prevent division by zero
                Vdenominator[i, j] += gamma * V[i, j] + (beta / 2) / sqrtV

        # Update V using multiplicative update rule
        for i in prange(M, nogil=True):
            for j in range(P):
                if Vdenominator[i, j] != 0:
                    V[i, j] *= Vnumerator[i, j] / (Vdenominator[i, j] + 1e-16)

        # Ensure non-negativity
        for i in prange(M, nogil=True):
            for j in range(P):
                if V[i, j] < 0:
                    V[i, j] = 0