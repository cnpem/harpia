# cython: boundscheck=False, wraparound=False
import numpy as np
from skimage.feature import local_binary_pattern
from time import time

cimport numpy as np
cimport cython
from cython cimport boundscheck, wraparound, parallel
from harpia.common import Size

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
#---------------------------------------------------------------------------------------------------

@boundscheck(False)
@wraparound(False)
def pixel_feature_extract(np.ndarray[numeric, ndim=3] hostImage,
                           list sigmas,
                           bint use_3d,
                           bint Intensity=True,
                           bint Edges=True,
                           bint Texture=False,
                           int verbose=0,
                           float gpuMemory=0.4,
                           int ngpus=-1):

    isize = Size(hostImage)
    cdef int z, feature_index

    cdef int feats_per_sigma = Intensity + Edges + Texture
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
        
        # 3D LBP is not directly implemented. 2D is available in scikit-image. Run per slice.
        if Texture:
            for z in range(isize.z):
                    lbp = local_binary_pattern(blurred_3d[z], P=8, R=1, method='uniform')
                    results[feature_index, z] = lbp.astype(np.float32)
            feature_index += 1
                
    print("\n Feature extraction completed in {:.2f} seconds.\n".format(time() - start_time))

    return results