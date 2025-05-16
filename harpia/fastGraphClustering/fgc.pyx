import numpy as np
cimport numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from cython.parallel import prange
from scipy.ndimage import uniform_filter
from cython cimport boundscheck, wraparound, parallel
from time import perf_counter

ctypedef fused numeric:
    int
    unsigned int
    float

# Disable bounds checking and wraparound for performance
@boundscheck(False)
@wraparound(False)
def solver_fgc(np.ndarray[np.float32_t, ndim=2] Z,
               np.ndarray[np.float32_t, ndim=2] Y,
               float lambda_,
               int iterations,
               float tolerance):
    """
    Solves the minimization problem:
        min_{Y} Tr(Y^{T}LY)
        s.t. Y^{T}Y = I, Y >= 0
    using a multiplicative update rule and a anchor graph approach (L=ZRZ^T).

    Parameters:
    ----------
    Z: np.ndarray, shape (P, M)
    Y: np.ndarray, shape (P, M) (Modified in place)
    lambda_: float
    iterations: int
    tolerance: float
    """
    cdef int P = Y.shape[0]  # Number of pixels
    cdef int M = Y.shape[1]  # Number of representative points

    # Pre-compute auxiliary variables
    cdef np.ndarray[np.float32_t, ndim=1] D = np.sum(Z, axis=0).astype(np.float32)  # Z.colwise().sum()
    cdef np.ndarray[np.float32_t, ndim=2] ZD = np.zeros_like(Z, dtype=np.float32)

    # Compute ZD = Z @ diag(D)
    cdef int i, j, k
    for i in prange(P, nogil=True):
        for j in range(M):
            ZD[i, j] = Z[i, j] * D[j]

    # Initialize matrices
    cdef np.ndarray[np.float32_t, ndim=2] ZtY = np.empty((M, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] YtY = np.empty((M, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Numerator = np.empty((P, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Denominator = np.empty((P, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Yold = np.copy(Y)

    cdef float currentObjective

    # Iterative update
    for _ in range(iterations):

        # Compute Z^T * Y
        for i in prange(M, nogil=True):
            for j in range(M):
                ZtY[i, j] = 0
                for k in range(P):
                    ZtY[i, j] += Z[k, i] * Y[k, j]

        # Compute Y^T * Y
        for i in prange(M, nogil=True):
            for j in range(M):
                YtY[i, j] = 0
                for k in range(P):
                    YtY[i, j] += Y[k, i] * Y[k, j]

        # Compute Numerator and denominator
        for i in prange(P, nogil=True):
            for j in range(M):
                Numerator[i, j] = 2 * lambda_ * Y[i, j]
                Denominator[i, j] = Y[i, j]
                for k in range(M):
                    Numerator[i, j] += ZD[i, k] * ZtY[k, j]
                    Denominator[i, j] += 2 * lambda_ * Y[i, k] * YtY[k, j]

        # Update Y
        for i in prange(P, nogil=True):
            for j in range(M):
                if Denominator[i, j] != 0:
                    Y[i, j] *= Numerator[i, j] / Denominator[i, j]

        # Compute Frobenius norm
        currentObjective = 0.0
        for i in prange(P, nogil=True):
            for j in range(M):
                currentObjective += (Y[i, j] - Yold[i, j]) ** 2
        currentObjective = currentObjective ** 0.5

        # Check for convergence
        if currentObjective < tolerance:
            print(f"Converged in iteration {_}")
            break

        # Update Yold
        for i in prange(P, nogil=True):
            for j in range(M):
                Yold[i, j] = Y[i, j]


    return Y


@boundscheck(False)
@wraparound(False)
def solver_fgc_with_labels(np.ndarray[np.float32_t, ndim=2] Z,
               np.ndarray[np.float32_t, ndim=2] Y,
               np.ndarray[np.float32_t, ndim=2] F,
               float lambda_,
               float gamma,
               int iterations,
               float tolerance):
    """
    Solves the minimization problem:
        min_{Y} Tr(Y^{T}LY)
        s.t. Y^{T}Y = I, Y >= 0
    using a multiplicative update rule and a anchor graph approach (L=ZRZ^T).

    Parameters:
    ----------
    Z: np.ndarray, shape (P, M)
    Y: np.ndarray, shape (P, M) (Modified in place)
    lambda_: float
    iterations: int
    tolerance: float
    """
    cdef int P = Y.shape[0]  # Number of pixels
    cdef int M = Y.shape[1]  # Number of representative points

    # Pre-compute auxiliary variables
    cdef np.ndarray[np.float32_t, ndim=1] D = np.sum(Z, axis=0).astype(np.float32)  # Z.colwise().sum()
    cdef np.ndarray[np.float32_t, ndim=2] ZD = np.zeros_like(Z, dtype=np.float32)

    # Compute ZD = Z @ diag(D)
    cdef int i, j, k
    for i in prange(P, nogil=True):
        for j in range(M):
            ZD[i, j] = Z[i, j] * D[j]

    # Initialize matrices
    cdef np.ndarray[np.float32_t, ndim=2] ZtY = np.empty((M, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] YtY = np.empty((M, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Numerator = np.empty((P, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Denominator = np.empty((P, M), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Yold = np.copy(Y)

    cdef float currentObjective

    # Iterative update
    for _ in range(iterations):

        # Compute Z^T * Y
        for i in prange(M, nogil=True):
            for j in range(M):
                ZtY[i, j] = 0
                for k in range(P):
                    ZtY[i, j] += Z[k, i] * Y[k, j]

        # Compute Y^T * Y
        for i in prange(M, nogil=True):
            for j in range(M):
                YtY[i, j] = 0
                for k in range(P):
                    YtY[i, j] += Y[k, i] * Y[k, j]

        # Compute Numerator and denominator
        for i in prange(P, nogil=True):
            for j in range(M):
                Numerator[i, j] = 2 * lambda_ * Y[i, j] +  + gamma * F[i, j]
                Denominator[i, j] = (1 + gamma) * Y[i, j]
                for k in range(M):
                    Numerator[i, j] += ZD[i, k] * ZtY[k, j]
                    Denominator[i, j] += 2 * lambda_ * Y[i, k] * YtY[k, j]

        # Update Y
        for i in prange(P, nogil=True):
            for j in range(M):
                if Denominator[i, j] != 0:
                    Y[i, j] *= Numerator[i, j] / Denominator[i, j]

        # Compute Frobenius norm
        currentObjective = 0.0
        for i in prange(P, nogil=True):
            for j in range(M):
                currentObjective += (Y[i, j] - Yold[i, j]) ** 2
        currentObjective = currentObjective ** 0.5

        # Check for convergence
        if currentObjective < tolerance:
            print(f"Converged in iteration {_}")
            break

        # Update Yold
        for i in prange(P, nogil=True):
            for j in range(M):
                Yold[i, j] = Y[i, j]


    return Y

class general_fgc:
    """
    FGC (Fast Graph Clustering) Class

    This class implements the Fast Graph Clustering algorithm for image segmentation.

    Parameters:
        - x (np.ndarray): Input data.
        - basis (np.ndarray, optional): Reference material for segmentation. Default is None.
        - lmbd (float, optional): Regularization parameter. Default is 1.0.
        - k (int, optional): Number of anchors. Default is 4.
        - iterations (int, optional): Number of iterations for the optimization algorithm. Default is 1000.
        - tol (float, optional): Tolerance for convergence. Default is 1e-2.
        - metric (str, optional): Metric for computing similarity between points ('euclidean' by default).
        - anchor_finder (str, optional): Method for finding anchor points ('kmeans' or 'random'). Default is 'kmeans'.
        - beta (float, optional): Weight for the spatial term. Default is 0.0.

    Methods:
        - anchor(): Computes the anchor matrix based on the chosen anchor finding method.
        - classification(): Calls the C++ implementation of the FGC algorithm.

    Attributes:
        - x (np.ndarray): Input data matrix.
        - u (np.ndarray): Reference material matrix.
        - y (np.ndarray): Random matrix.
        - metric (str): Metric for similarity.
        - k (int): Number of anchors.
        - anchor_finder (str): Method for finding anchor points.
        - z (np.ndarray): Anchor matrix.
        - lmbd (float): Regularization parameter.
        - iterations (int): Number of iterations for optimization.
        - tol (float): Tolerance for convergence.
        - beta (float): Weight for the spatial term.

    Usage:
        - fgc_instance = general_fgc(x, basis, lmbd, k, iterations, tol, metric, anchor_finder, beta)
        - fgc_instance.classification()

        - After calling classification, apply k-means clustering to the resulting y matrix:
        - kmeans = KMeans(n_clusters=your_desired_number_of_clusters)  # Replace 'your_desired_number_of_clusters' with the actual number
        - clusters = kmeans.fit_predict(fgc_instance.y)

        - 'clusters' now contains the cluster assignments for each data point in the y matrix.

    Note:
        - The FGC class relies on a C++ implementation (multiplicativeUpdate) for the main algorithm. 
        - After calling the 'classification' method, you can use k-means clustering on the 'y' matrix 
        - to obtain cluster assignments for further analysis.
    """

    def __init__(
        self,
        x: np.ndarray,
        rows: int,
        cols: int,
        basis: np.ndarray = None,
        lmbd: float = 1.0,
        k: int = 4,
        iterations: int = 1000,
        tol: float = 1e-2,
        metric: str = 'euclidean',
        anchor_finder: str = 'kmeans',
        beta: float = 0.0,
        size: int = 3
    ) -> None:
        # Image phase space
        self.x = x
        
        # Reference materials
        self.u = basis

        # Random matrix
        self.y = np.random.rand(self.x[0].size, k)

        # Metric for similarity
        self.metric = metric

        # Number of anchors
        self.k = k

        # Anchor type
        self.anchor_finder = anchor_finder

        # Regularization
        self.lmbd = lmbd

        # Number of iterations
        self.iterations = iterations

        # Tolerance
        self.tol = tol

        # Weight for spatial term
        self.beta = beta

        # Window size for spatial term
        self.size = size

        self.cols = cols
        self.rows= rows

        # Anchor matrix
        self.z = self.anchor(self.beta)

    def anchor(self, beta: float) -> np.ndarray:
        """
        Compute the anchor matrix based on the chosen method and spatial term weight.

        Parameters:
        - beta (float): Weight for the spatial term.

        Returns:
        - np.ndarray: Anchor matrix.
        """
        
        if self.u is None:
            if self.anchor_finder == 'random':
                random_index1 = np.random.randint(0, self.x.shape[1], size=self.k)
                self.u = self.x[:, random_index1]
            elif self.anchor_finder == 'kmeans':
                self.u = KMeans(n_clusters=self.k, n_init="auto").fit(self.x.T).cluster_centers_.T
            else:
                raise ValueError("Invalid value for anchor_finder. Supported values are 'random' and 'kmeans'.")    

        # Distance matrix (spectral distance)
        dis_spectral = distance.cdist(self.x.T, self.u.T, self.metric)
        x_reshaped = self.x.reshape((-1, self.rows, self.cols))

        if beta > 0:
            # Compute the spatial term
            window_size = self.size  # Example window size for neighboring pixels
            spatial_means_2d = uniform_filter(x_reshaped, size=(1, window_size, window_size), mode='reflect')
            
            spatial_means = spatial_means_2d.reshape(self.x.shape)
            
            # Distance matrix (spatial distance)
            dis_spatial = distance.cdist(spatial_means.T, self.u.T, self.metric)
            
            # Combine spectral and spatial distances
            dis = dis_spectral + beta * dis_spatial
        else:
            dis = dis_spectral
        
        # Number of anchors
        numNearestAnchor = min(self.k, dis.shape[1])
        
        # Sort the distances and get the indices
        idx = np.argsort(dis, axis=1)
        
        # Anchor matrix allocation
        anchor_num, num = self.u.shape[1], self.x.shape[1]
        A = np.zeros((num, anchor_num), dtype=np.float32)
        
        # Compute linear regression for anchors
        numCols = min(numNearestAnchor + 1, dis.shape[1])
        for i in range(num):
            idx_i = idx[i, :numCols]
            di = dis[i, idx_i]
            A[i, idx_i] = (di[numCols - 1] - di) / (numNearestAnchor * di[numNearestAnchor - 1] - np.sum(di[:numNearestAnchor]) + np.finfo(float).eps)
        
        return A

    def classification(self, labels: np.ndarray=None, gamma: float=0) -> None:
        """
        Calls the C++ implementation of the FGC algorithm for image segmentation.

        Returns:
        - None
        """
        
        # All matrices must be in the C++ format (Fortran/column-major)
        # they also must have the correct data type, i.e., float32
        self.z = np.asfortranarray(self.z, dtype=np.float32)
        self.y = np.asfortranarray(self.z.copy(), dtype=np.float32)

        # Calls C++ implementation
        if labels is None:
            solver_fgc(self.z, self.y, self.lmbd, self.iterations, self.tol)
        else:
            solver_fgc_with_labels(self.z, self.y, labels, self.lmbd, gamma, self.iterations, self.tol)
