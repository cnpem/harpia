import cupy as cp
import numpy as np                     # For array manipulation

#----------------------------------------
# Skimage functions
#----------------------------------------
from skimage.morphology import (
    binary_erosion, 
    binary_dilation, 
    binary_closing, 
    binary_opening
)
from skimage.morphology import (
    erosion, 
    dilation, 
    closing, 
    opening,
    white_tophat, 
    black_tophat, 
    reconstruction
)
from skimage.filters import (
    prewitt,
    sobel,
    gaussian,
    threshold_niblack,
    threshold_sauvola,
    threshold_mean,
    #threshold_gaussian
)

#----------------------------------------
# cuCIM functions
#----------------------------------------
from cucim.skimage import morphology as cucim_morph
from cucim.skimage import filters as cucim_filters


#----------------------------------------
# Harpia functions
#----------------------------------------

# Workaround to allow importing harpia python module locally
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

# Harpia functions
from harpia.morphology.operations_binary import (
     erosion_binary,
     dilation_binary,
     closing_binary,
     opening_binary,
     smooth_binary,
     geodesic_erosion_binary,
     geodesic_dilation_binary,
     reconstruction_binary,
     fill_holes
)

from harpia.morphology.operations_grayscale import (
     erosion_grayscale,
     dilation_grayscale,
     closing_grayscale,
     opening_grayscale,
     geodesic_erosion_grayscale,
     geodesic_dilation_grayscale,
     reconstruction_grayscale,
     top_hat,
     bottom_hat,
     top_hat_reconstruction,
     bottom_hat_reconstruction,
)

from harpia.filters.filtersChunked import (
     gaussianFilter,
     meanFilter,
     logFilter,
     unsharpMaskFilter,
     sobelFilter,
     prewittFilter,
     anisotropic_diffusion3D,
)

from harpia.threshold.thresholdChunked import(
    gaussianThreshold,
    meanThreshold,
    niblackThreshold,
    sauvolaThreshold
)


import harpia
print(harpia.__file__)

#----------------------------------------
# Workaround functions
#----------------------------------------

def custum_kernel3D():
    kernel_2d = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
    # Stack the 2D kernel to form a 3D kernel (3 layers)
    kernel_3d = np.stack([kernel_2d, kernel_2d, kernel_2d])
    return kernel_3d

def smooth_sk(image, selem):
    result = opening(image, selem)
    result = binary_closing(result, selem)
    return result

def smooth_cucim(image, selem):
    image = cp.asarray(image)
    selem = cp.asarray(selem)
    result = cucim_morph.binary_opening(image, selem)
    result = cucim_morph.binary_closing(result, selem)
    return result

#----------------------------------------
# Dict for tests
#----------------------------------------

kernel = custum_kernel3D()

morphology_grayscale = [
    {
        "name": "Erosion 3D grayscale",
        "skimage": erosion,
        "custom": erosion_grayscale,
        "cucim": cucim_morph.erosion,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Dilation 3D grayscale",
        "skimage": dilation,
        "custom": dilation_grayscale,
        "cucim": cucim_morph.dilation,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Closing 3D grayscale",
        "skimage": closing,
        "custom": closing_grayscale,
        "cucim": cucim_morph.closing,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Opening 3D grayscale",
        "skimage": opening,
        "custom": opening_grayscale,
        "cucim": cucim_morph.opening,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Top Hat 3D grayscale",
        "skimage": white_tophat,
        "custom": top_hat,
        "cucim": cucim_morph.white_tophat,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Bottom Hat 3D grayscale",
        "skimage": black_tophat,
        "custom": bottom_hat,
        "cucim": cucim_morph.black_tophat,
        "kernel":kernel,
        "multi-gpu": True
    },
]

morphology_binary = [
    {
        "name": "Erosion 3D binary",
        "skimage": erosion,
        "custom": erosion_binary,
        "cucim": cucim_morph.binary_erosion,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Dilation 3D binary",
        "skimage": dilation,
        "custom": dilation_binary,
        "cucim": cucim_morph.binary_dilation,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Closing 3D binary",
        "skimage": closing,
        "custom": closing_binary,
        "cucim": cucim_morph.binary_closing,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Opening 3D binary",
        "skimage": opening,
        "custom": opening_binary,
        "cucim": cucim_morph.binary_opening,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Smoothing 3D binary",
        "skimage": smooth_sk,
        "custom": smooth_binary,
        "cucim": smooth_cucim,
        "kernel":kernel,
        "multi-gpu": True
    },
]

operations_filters = [
    {
        "name": "Gaussian Filter 3D grayscale",
        "skimage": gaussian,
        "skimage_param": {'mode': 'reflect'},
        "custom": gaussianFilter,
        "cucim": cucim_filters.gaussian,
        "kernel":None,
        "multi-gpu": True
    },
    {
        "name": "Mean Filter 3D grayscale",
        # "skimage": None,
        "custom": meanFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True  # 2D only – no 3D support in cucim
    },
    {
        "name": "Log Filter 3D grayscale",
        "skimage": None,
        "custom": logFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True  
    },
    {
        "name": "Unsharp Mask Filter 3D grayscale",
        "skimage": None,
        "custom": unsharpMaskFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True   # Not available in cucim as of now
    },
    #{
    #    "name": "Non local means Filter 3D grayscale",
    #    "skimage": None,
    #    "custom": non_local_means,
    #    "cucim": None  # No NLM in cucim
    #},
    {
        "name": "Sobel Filter 3D grayscale",
        "skimage": sobel,
        "custom": sobelFilter,
        "cucim": cucim_filters.sobel,
        "kernel":None,
        "multi-gpu": True 
    },
    {
        "name": "Prewitt Filter 3D grayscale",
        "skimage": prewitt,
        "custom": prewittFilter,
        "cucim": cucim_filters.prewitt,
        "kernel":None,
        "multi-gpu": True 
    },
    # {
    #     "name": "Median Filter 3D grayscale",
    #     "skimage": None,
    #     "custom": median,
    #     "cucim": cucim_filters.median,
    #     "kernel":None   # 2D support; 3D needs manual handling
    # },
    {
        "name": "Anisotropic Diffusion Filter 3D grayscale",
        "skimage": None,
        "custom": anisotropic_diffusion3D,
        "cucim": None, # Not implemented in cucim
        "kernel":None,
        "multi-gpu": True   
    },
]

operations_thresholds = [
    {
        "name": "Threshold Niblack",
        "skimage": threshold_niblack,
        "custom": niblackThreshold,  
        "cucim": cucim_filters.threshold_niblack,
        "kernel":None 
    },
    {
        "name": "Threshold Sauvola",
        "skimage": threshold_sauvola,
        "custom": sauvolaThreshold,
        "cucim": cucim_filters.threshold_sauvola,
        "kernel":None 
    },
    {
        "name": "Threshold Mean",
        "skimage": threshold_mean,
        "custom": meanThreshold,
        "cucim": None,
        "kernel":None 
    },
    #{
    #    "name": "Threshold Gaussian",
    #    "skimage": threshold_gaussian,
    #    "custom": gaussianThreshold,
    #    "cucim": lambda img, **kwargs: cucim_threshold_local(img, method='gaussian', **kwargs)
    #}
]

grayscale = operations_thresholds + morphology_grayscale + operations_filters
grayscale_no_threashold = morphology_grayscale + operations_filters

binary  = morphology_binary

def filter_operations_by_framework(operations, keep_key):
    """Returns a new list of operations keeping only one framework function,
    and includes only those where keep_key is not None.
    """
    other_keys = {"skimage", "custom", "cucim"} - {keep_key}
    new_ops = []
    for op in operations:
        if op.get(keep_key) is not None:
            op_filtered = op.copy()
            for key in other_keys:
                op_filtered[key] = None
            new_ops.append(op_filtered)
    return new_ops

# Create 3 framework-specific versions
grayscale_skimage = filter_operations_by_framework(grayscale, "skimage")
grayscale_custom  = filter_operations_by_framework(grayscale, "custom")
grayscale_custom_morphology  = filter_operations_by_framework(morphology_grayscale, "custom")
grayscale_custom_threashold  = filter_operations_by_framework(operations_thresholds, "custom")
grayscale_custom_filters  = filter_operations_by_framework(operations_filters, "custom")

grayscale_cucim   = filter_operations_by_framework(grayscale, "cucim")
grayscale_cucim_no_threashold = filter_operations_by_framework(grayscale_no_threashold, "cucim")

binary_skimage = filter_operations_by_framework(binary, "skimage")
binary_custom  = filter_operations_by_framework(binary, "custom")
binary_cucim   = filter_operations_by_framework(binary, "cucim")
