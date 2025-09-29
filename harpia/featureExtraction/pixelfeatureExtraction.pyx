# cython: boundscheck=False, wraparound=False
import numpy as np
from time import time

cimport numpy as np
cimport cython
from cython cimport boundscheck, wraparound, parallel
from harpia.common import Size

#---------------------------------------------------------------------------------------------------
# Declare external CUDA feature extraction function
cdef extern from "../../include/superpixelExtraction/device_feat_extraction.h":
    void DeviceFeatExtraction2D(float* hostImage, float* hostOutput,
    int xsize, int ysize, int zsize,
    int nfeatures,
    float* sigmas,
    int nsigmas,
    bint intensity,
    bint edges,
    bint texture,
    bint shapeIndex,
    bint localBinaryPattern,
    int verbose, 
    float gpuMemory, 
    int ngpus)

#---------------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def pixel_feature_extract(np.ndarray[np.float32_t, ndim=3] hostImage,
                          np.ndarray[np.float32_t, ndim=1] sigmas,
                          dict features=None,
                          bint use_3d=False,
                          int verbose=0,
                          float gpuMemory=0.4,
                          int ngpus=-1):

    cdef double start_time = time()

    cdef isize = Size(hostImage)
    cdef int nsigmas = sigmas.shape[0]
    cdef int feats_per_sigma, total_features

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

    feats_per_sigma = Intensity + Edges + 2 * Texture + ShapeIndex + LocalBinaryPattern
    total_features = feats_per_sigma * nsigmas

    print("Sigmas len:", nsigmas)
    print("Total features:", total_features)

    # Output feature image: shape (nfeatures, z, y, x)
    cdef np.ndarray[np.float32_t, ndim=4] hostOutput = np.zeros((total_features, isize.z, isize.y, isize.x), dtype=np.float32)

    DeviceFeatExtraction2D(
        &hostImage[0, 0, 0],
        &hostOutput[0, 0, 0, 0],
        isize.x, isize.y, isize.z,
        total_features,
        &sigmas[0],
        nsigmas,
        Intensity,
        Edges,
        Texture,
        ShapeIndex,
        LocalBinaryPattern,
        verbose,
        gpuMemory,
        ngpus
    )

    print("\nFeature extraction completed in {:.2f} seconds.\n".format(time() - start_time))

    return hostOutput
