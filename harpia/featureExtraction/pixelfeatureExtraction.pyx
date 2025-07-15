# cython: boundscheck=False, wraparound=False
import numpy as np
from time import time

cimport numpy as np
cimport cython
from cython cimport boundscheck, wraparound, parallel
from harpia.common import Size
from harpia.shapeIndex.shape_index import hessian_eigenvalues, shape_index
#---------------------------------------------------------------------------------------------------
#Define the fused type for numeric types : float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int

#---------------------------------------------------------------------------------------------------
cdef extern from "../../include/filters/gaussian_filter.h":
    void gaussianFilterChunked[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, 
                                        int zsize, float sigma, int type3d, int verbose, int ngpus, 
                                        float gpuMemory)

cdef extern from "../../include/filters/prewitt_filter.h":
    void prewittFilterChunked[numeric](numeric* hostImage, float* hostOutput,
                                       int xsize, int ysize, int zsize, int type3d,
                                       int flag_verbose, int ngpus, float gpuMemory)

cdef extern from "../../include/localBinaryPattern/lbp.h":
    void localBinaryPattern[numeric](numeric* hostImage, float* hostOutput, int xsize, int ysize, int zsize)
#---------------------------------------------------------------------------------------------------

@boundscheck(False)
@wraparound(False)
def pixel_feature_extract(np.ndarray[numeric, ndim=3] hostImage,
                          list sigmas,
                          bint use_3d,
                          dict features=None,
                          int verbose=0,
                          float gpuMemory=0.4,
                          int ngpus=-1):

    isize = Size(hostImage)
    cdef int z, feature_index

    if features is None:
        features = {
            "Intensity": True,
            "Edges": True,
            "Texture": True,
            "ShapeIndex": False,
            "LocalBinaryPattern": False,
        }

    cdef bint Intensity = features.get("Intensity", False)
    cdef bint Edges = features.get("Edges", False)
    cdef bint Texture = features.get("Texture", False)
    cdef bint ShapeIndex = features.get("ShapeIndex", False)
    cdef bint LocalBinaryPattern = features.get("LocalBinaryPattern", False)

    cdef int feats_per_sigma = Intensity + Edges + 2 * Texture + LocalBinaryPattern + ShapeIndex
    print("Sigmas len", len(sigmas))
    cdef int total_features = len(sigmas) * feats_per_sigma
    print("Total features:", total_features)
    cdef np.ndarray[np.float32_t, ndim=4] results = np.zeros((total_features, isize.z, isize.y, isize.x), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] blurred_3d = np.zeros((isize.z, isize.y, isize.x), dtype=np.float32)

    feature_index = 0
    print("Running feature extraction.")
    start_time = time()
    for sigma in sigmas:
        gaussianFilterChunked(&hostImage[0, 0, 0], &blurred_3d[0, 0, 0],
                        isize.y, isize.x, isize.z,
                        float(sigma), use_3d, verbose, ngpus, gpuMemory)

        if Intensity:
            results[feature_index] = blurred_3d
            feature_index += 1

        if Edges:
            prewittFilterChunked(&blurred_3d[0, 0, 0], &results[feature_index, 0, 0, 0], isize.y, isize.x, isize.z, use_3d,
                    verbose, ngpus, gpuMemory)
            feature_index += 1

        eigenvalues = None
        if Texture:
            eigenvalues = hessian_eigenvalues(blurred_3d, step=1, verbose=0, gpuMemory=0.4, ngpus=-1)
            results[feature_index, :, :, :] = eigenvalues[:, :, :, 0]
            results[feature_index + 1, :, :, :] = eigenvalues[:, :, :, 1]
            feature_index += 2

        if ShapeIndex:
            results[feature_index, :] = shape_index(blurred_3d, eigen = eigenvalues,step=1, verbose=0, gpuMemory=0.4, ngpus=-1)
            feature_index += 1

        if LocalBinaryPattern:
            localBinaryPattern(&blurred_3d[0, 0, 0], &results[feature_index, 0, 0, 0], isize.y, isize.x, isize.z)
            feature_index += 1

    print("\n Feature extraction completed in {:.2f} seconds.\n".format(time() - start_time))

    return results