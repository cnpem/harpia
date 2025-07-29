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
    threshold_local
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
    result = binary_opening(image, selem)
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
        "harpia": erosion_grayscale,
        "cucim": cucim_morph.erosion,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Dilation 3D grayscale",
        "skimage": dilation,
        "harpia": dilation_grayscale,
        "cucim": cucim_morph.dilation,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Closing 3D grayscale",
        "skimage": closing,
        "harpia": closing_grayscale,
        "cucim": cucim_morph.closing,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Opening 3D grayscale",
        "skimage": opening,
        "harpia": opening_grayscale,
        "cucim": cucim_morph.opening,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Top Hat 3D grayscale",
        "skimage": white_tophat,
        "harpia": top_hat,
        "cucim": cucim_morph.white_tophat,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Bottom Hat 3D grayscale",
        "skimage": black_tophat,
        "harpia": bottom_hat,
        "cucim": cucim_morph.black_tophat,
        "kernel":kernel,
        "multi-gpu": True
    },
]

morphology_binary = [
    {
        "name": "Erosion 3D binary",
        "skimage": binary_erosion,
        "harpia": erosion_binary,
        "cucim": cucim_morph.binary_erosion,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Dilation 3D binary",
        "skimage": binary_dilation,
        "harpia": dilation_binary,
        "cucim": cucim_morph.binary_dilation,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Closing 3D binary",
        "skimage": binary_closing,
        "harpia": closing_binary,
        "cucim": cucim_morph.binary_closing,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Opening 3D binary",
        "skimage": binary_opening,
        "harpia": opening_binary,
        "cucim": cucim_morph.binary_opening,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Smoothing 3D binary",
        "skimage": smooth_sk,
        "harpia": smooth_binary,
        "cucim": smooth_cucim,
        "kernel":kernel,
        "multi-gpu": True
    },
]



operations_filters = [
    {
        "name": "Anisotropic Diffusion Filter 3D grayscale",
        "skimage": None,
        "harpia": anisotropic_diffusion3D,
        "cucim": None, # Not implemented in cucim
        "kernel":None,
        "multi-gpu": True   
    },
    {
        "name": "Gaussian Filter 3D grayscale",
        "skimage": gaussian,
        "skimage_param": {'mode': 'reflect'},
        "harpia": None, #gaussianFilter,
        "cucim": cucim_filters.gaussian,
        "kernel":None,
        "multi-gpu": True
    },
    #{
    #    "name": "Non local means Filter 3D grayscale",
    #    "skimage": None,
    #    "harpia": non_local_means,
    #    "cucim": None  # No NLM in cucim
    #},
    {
        "name": "Sobel Filter 3D grayscale",
        "skimage": sobel,
        "harpia": sobelFilter,
        "cucim": cucim_filters.sobel,
        "kernel":None,
        "multi-gpu": True 
    },

    # {
    #     "name": "Median Filter 3D grayscale",
    #     "skimage": None,
    #     "harpia": median,
    #     "cucim": cucim_filters.median,
    #     "kernel":None   # 2D support; 3D needs manual handling
    # },
        {
        "name": "Mean Filter 3D grayscale",
        "skimage": None,
        "harpia": meanFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True  # 2D only – no 3D support in cucim
    },
    {
        "name": "Prewitt Filter 3D grayscale",
        "skimage": prewitt,
        "harpia": prewittFilter,
        "cucim": cucim_filters.prewitt,
        "kernel":None,
        "multi-gpu": True 
    },
    {
        "name": "Log Filter 3D grayscale",
        "skimage": None,
        "harpia": logFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True  
    },
    {
        "name": "Unsharp Mask Filter 3D grayscale",
        "skimage": None,
        "harpia": None, #unsharpMaskFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True   # Not available in cucim as of now
    },
]

operations_thresholds = [
    {
        "name": "Threshold Niblack",
        "skimage": threshold_niblack,
        "harpia": niblackThreshold,  
        "cucim": cucim_filters.threshold_niblack,
        "kernel":None 
    },
    {
        "name": "Threshold Sauvola",
        "skimage": threshold_sauvola,
        "harpia": sauvolaThreshold,
        "cucim": cucim_filters.threshold_sauvola,
        "kernel":None 
    },
    {
        "name": "Threshold Mean",
        "skimage": threshold_mean,
        "harpia": meanThreshold,
        "cucim": None,
        "kernel":None 
    },
    {
       "name": "Threshold Gaussian",
       "skimage": threshold_local,
       "skimage_param": {'method': 'gaussian', 'block_size':7, 'mode':'reflect'},
       "harpia": None,# gaussianThreshold,
       "cucim": cucim_filters.threshold_local,
       "cucim_param": {'method': 'gaussian', 'block_size':7, 'mode':'reflect'},
       "kernel":None 
    }
]


grayscale = operations_filters + operations_thresholds + morphology_grayscale
binary  = morphology_binary

def filter_operations_by_framework(operations, keep_key):
    """Returns a new list of operations keeping only one framework function,
    and includes only those where keep_key is not None.
    """
    other_keys = {"skimage", "harpia", "cucim"} - {keep_key}
    new_ops = []
    for op in operations:
        if op.get(keep_key) is not None:
            op_filtered = op.copy()
            for key in other_keys:
                op_filtered[key] = None
            new_ops.append(op_filtered)
    return new_ops

# Create 3 framework-specific versions
skimage_binary = filter_operations_by_framework(binary, "skimage")
skimage_grayscale = filter_operations_by_framework(grayscale, "skimage")

harpia_binary  = filter_operations_by_framework(binary, "harpia")
harpia_grayscale  = filter_operations_by_framework(morphology_grayscale + operations_filters +
                                                   operations_thresholds, "harpia")

cucim_bianry   = filter_operations_by_framework(binary, "cucim")
cucim_grayscale   = filter_operations_by_framework(grayscale, "cucim")
cucim_grayscale_no_threashold = filter_operations_by_framework(morphology_grayscale + operations_filters, 
                                                               "cucim")

operations_cucim_grayscale_512_aida =[
    {
        "name": "Erosion 3D grayscale",
        "skimage": erosion,
        "harpia": erosion_grayscale,
        "cucim": cucim_morph.erosion,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Dilation 3D grayscale",
        "skimage": dilation,
        "harpia": dilation_grayscale,
        "cucim": cucim_morph.dilation,
        "kernel":kernel,
        "multi-gpu": True
    },
    {
        "name": "Gaussian Filter 3D grayscale",
        "skimage": gaussian,
        "skimage_param": {'mode': 'reflect'},
        "harpia": gaussianFilter,
        "cucim": cucim_filters.gaussian,
        "kernel":None,
        "multi-gpu": True
    },
]
cucim_grayscale_512_aida = filter_operations_by_framework(operations_cucim_grayscale_512_aida, 
                                                               "cucim")
operations_harpia_gauss = [
    {
       "name": "Threshold Gaussian",
       "skimage": None, #threshold_local,
       "skimage_param": {'method': 'gaussian', 'block_size':7, 'mode':'reflect'},
       "harpia": gaussianThreshold,
       "cucim": None, #cucim_filters.threshold_local,
       "cucim_param": {'method': 'gaussian', 'block_size':7, 'mode':'reflect'},
       "kernel":None 
    },
]
filters_harpia_gauss = [
    {
        "name": "Gaussian Filter 3D grayscale",
        "skimage": None, # gaussian,
        "skimage_param": {'mode': 'reflect'},
        "harpia": gaussianFilter,
        "cucim": None, # cucim_filters.gaussian,
        "kernel":None,
        "multi-gpu": True
    },
    {
        "name": "Unsharp Mask Filter 3D grayscale",
        "skimage": None,
        "harpia": unsharpMaskFilter,
        "cucim": None,
        "kernel":None,
        "multi-gpu": True   # Not available in cucim as of now
    },
]
harpia_gauss  = filter_operations_by_framework(operations_harpia_gauss+filters_harpia_gauss, "harpia")

filter_harpia  = filter_operations_by_framework(operations_filters, "harpia")
filter_harpia_gauss  = filter_operations_by_framework(filters_harpia_gauss, "harpia")
threashold_harpia_gauss  = filter_operations_by_framework(operations_harpia_gauss, "harpia")
