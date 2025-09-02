# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

from time import time

cimport cython
from cython cimport boundscheck, wraparound, parallel
from harpia.common import Size
from harpia.shapeIndex.shape_index import hessian_eigenvalues, shape_index

ctypedef fused numeric:
    float
    int
    unsigned int

cdef extern from "../../include/superpixelExtraction/pooling_superpixel.h":
    void DeviceSuperpixelPooling2D(float* hostImage,
        int* hostSuperPixel,
        float* hostOutput,
        int xsize, int ysize, int zsize,
        int nsuperpixels,
        int nfeatures,
        float* sigmas,
        int nsigmas,
        bint intensity,
        bint edges,
        bint texture,
        bint shapeIndex,
        bint localBinaryPattern,
        bint output_mean,
        bint output_min,
        bint output_max,
        int flag_verbose, 
        float gpuMemory, 
        int ngpus)

def superpixel_pooling_feature(np.ndarray[np.float32_t, ndim=3] hostImage,
                   np.ndarray[int, ndim=3] hostSuperPixelImage,
                   np.ndarray[np.float32_t, ndim=1] sigmas,
                   int numSuperpixels = -1,
                   np.ndarray[np.float32_t, ndim=2] hostOutput = None,
                   dict features=None,
                   int type3d = 1,
                   int verbose = 0, float gpuMemory = 0.4, int ngpus = -1):

    isize = Size(hostImage)

    if features is None:
        features = {
            "Intensity": True,
            "Edges": True,
            "Texture": True,
            "ShapeIndex": False,
            "LocalBinaryPattern": False,
            "pooling": {
                "output_mean": True,
                "output_min": False,
                "output_max": False
            }
        }

    cdef bint Intensity = features.get("Intensity", False)
    cdef bint Edges = features.get("Edges", False)
    cdef bint Texture = features.get("Texture", False)
    cdef bint ShapeIndex = features.get("ShapeIndex", False)
    cdef bint LocalBinaryPattern = features.get("LocalBinaryPattern", False)
    #get pooling options

    # Validate pooling options
    pooling = features.get("pooling", {})
    cdef bint output_mean = pooling.get("output_mean", False)
    cdef bint output_max = pooling.get("output_max", False)
    cdef bint output_min = pooling.get("output_min", False)
    if not (output_mean or output_max or output_min):
        raise ValueError("At least one pooling method must be enabled")


    if numSuperpixels == -1:
        numSuperpixels = int(np.max(hostSuperPixelImage) - np.min(hostSuperPixelImage) + 1)

    cdef int feats_per_sigma = Intensity + Edges + 2 * Texture + LocalBinaryPattern + ShapeIndex
    nsigmas = int(len(sigmas))
    print("Sigmas len", nsigmas)
    cdef int total_features = nsigmas * feats_per_sigma * (output_max + output_mean + output_min)
    print("Total features:", total_features)

    if hostOutput is None:
        hostOutput = np.empty((numSuperpixels, total_features), dtype=np.float32)

    DeviceSuperpixelPooling2D(
        &hostImage[0, 0, 0],
        &hostSuperPixelImage[0, 0, 0],
        &hostOutput[0, 0],
        isize.x, isize.y, isize.z,
        numSuperpixels,
        total_features,
        &sigmas[0],
        nsigmas,
        Intensity,
        Edges,
        Texture,
        ShapeIndex,
        LocalBinaryPattern,
        output_mean,
        output_min,
        output_max,
        verbose, 
        gpuMemory, 
        ngpus
    )

    return hostOutput
