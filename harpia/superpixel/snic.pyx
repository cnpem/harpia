import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, parallel

ctypedef fused numeric:
    int
    unsigned int
    float


cdef extern from "../../include/superpixel/snic.h":
    void snic_grayscale_heap(const float* image, int width, int height,
                             float spacing, int* labels, float m)
    void snic_grayscale_heap_2d_batched(const float* image, int width, int height,
                                    float spacing, int* labels, float m, int batch_size)
    void snic_grayscale_heap_3d(const float* image, int width, int height, int depth,
                                float spacing, int* labels, float m)
    void snic_grayscale_heap_3d_batched(const float* image, int width, int height, int depth,
                                        float spacing, int* labels, float m, int dz)
                    


def SNIC2D(np.ndarray[float, ndim=2] hostImage,
           np.ndarray[int, ndim=2] hostLabels = None,
           float spacing = 5, float m = 10.0):
    """
    Apply SNIC superpixel segmentation on a 2D grayscale image.

    Parameters:
        hostImage (ndarray): Input 2D image (float32).
        hostLabels (ndarray, optional): Output label map (int32). Auto-created if None.
        spacing (float): Initial spacing between superpixel seeds.
        m (float): Compactness parameter (controls color vs. spatial proximity).

    Returns:
        ndarray: Superpixel label map (int32).
    """
    cdef int height = hostImage.shape[0]
    cdef int width = hostImage.shape[1]

    if hostLabels is None:
        hostLabels = np.empty((height, width), dtype=np.int32)

    snic_grayscale_heap(&hostImage[0, 0],
                        width, height,
                        spacing, &hostLabels[0, 0], m)

    return hostLabels


def SNIC2DBatches(np.ndarray[np.float32_t, ndim=3] hostImage,
                  np.ndarray[int, ndim=3] hostLabels = None,
                  float spacing = 5, float m = 10.0, int batch_size = 8):
    """
    Apply SNIC to multiple 2D slices at once (batched along axis 0).

    Parameters:
        hostImage (ndarray): Input 3D image (num_slices × height × width), float32.
        hostLabels (ndarray, optional): Output label array (int32).
        spacing (float): Superpixel spacing.
        m (float): Compactness parameter.
        batch_size (int): Number of slices to process in each batch.

    Returns:
        ndarray: Superpixel labels (int32).
    """
    cdef int num_slices = hostImage.shape[0]
    cdef int height = hostImage.shape[1]
    cdef int width = hostImage.shape[2]

    if hostLabels is None:
        hostLabels = np.empty((num_slices, height, width), dtype=np.int32)

    snic_grayscale_heap_2d_batched(&hostImage[0, 0, 0],
                                   width, height,
                                   spacing, &hostLabels[0, 0, 0], m, batch_size)

    return hostLabels


def SNIC3D(np.ndarray[float, ndim=3] hostImage,
           np.ndarray[int, ndim=3] hostLabels = None,
           float spacing = 5, float m = 10.0):
    """
    Apply SNIC superpixel segmentation on a 3D grayscale volume.

    Parameters:
        hostImage (ndarray): Input 3D image (float32).
        hostLabels (ndarray, optional): Output label volume (int32). Auto-created if None.
        spacing (float): Initial spacing between supervoxel seeds.
        m (float): Compactness parameter.

    Returns:
        ndarray: Supervoxel label volume (int32).
    """
    cdef int depth = hostImage.shape[0]
    cdef int height = hostImage.shape[1]
    cdef int width = hostImage.shape[2]

    if hostLabels is None:
        hostLabels = np.empty((depth, height, width), dtype=np.int32)

    snic_grayscale_heap_3d(&hostImage[0, 0, 0],
                           width, height, depth,
                           spacing, &hostLabels[0, 0, 0], m)

    return hostLabels


def SNIC3DBatches(np.ndarray[float, ndim=3] hostImage,
                  np.ndarray[int, ndim=3] hostLabels = None,
                  float spacing = 5, float m = 10.0, int dz = 8):
    """
    Apply batched SNIC superpixel segmentation on a 3D grayscale volume.

    Parameters:
        hostImage (ndarray): Input 3D image (float32).
        hostLabels (ndarray, optional): Output label volume (int32). Auto-created if None.
        spacing (float): Initial spacing between supervoxel seeds.
        m (float): Compactness parameter.
        dz (int): Number of slices per batch along the z-axis.

    Returns:
        ndarray: Supervoxel label volume (int32).
    """
    cdef int depth = hostImage.shape[0]
    cdef int height = hostImage.shape[1]
    cdef int width = hostImage.shape[2]

    if hostLabels is None:
        hostLabels = np.empty((depth, height, width), dtype=np.int32)

    snic_grayscale_heap_3d_batched(&hostImage[0, 0, 0],
                                   width, height, depth,
                                   spacing, &hostLabels[0, 0, 0], m, dz)

    return hostLabels
