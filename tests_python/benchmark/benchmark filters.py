#!/usr/bin/env python3
import numpy as np                     # For array manipulation
import cupy as cp
from framework import image, tests

from skimage.filters import (
    prewitt,
    sobel,
    gaussian)

from skimage.filters import (
    threshold_niblack,
    threshold_sauvola,
    threshold_mean,
    #threshold_gaussian
)

from cucim.skimage import filters as cucim_filters
from cucim.skimage.filters import threshold_niblack as cucim_threshold_niblack
from cucim.skimage.filters import threshold_sauvola as cucim_threshold_sauvola
#from cucim.skimage.filters import threshold_mean as cucim_threshold_mean
from cucim.skimage.filters import threshold_local as cucim_threshold_local  # for gaussian-style (adaptive) thresholding

# Custom filters chunked operations from harpia
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

from harpia.filters.filters import median,non_local_means

import harpia
print(harpia.__file__)

#############
# Read Images
#############

# Instruction: Uncomment the image for which tests will be executed.

# IMAGE 1
#print("reading small image...")
#xsize = 190
#ysize = 207
#zsize_original = 100
#zsize = 100
#path_grayscale = "../../example_images/grayscale/crua_A_190x207x100_16b.raw"
#image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'uint16')
#img_num = 1
#print("finished reading small image!")

# # IMAGE 2 (possibily with problem)
# print("reading big image...")
# xsize = 2048
# ysize = 2048
# zsize = 1964
# path_grayscale = "../../../../../../../../beamlines/mogno/proposals/20180217/data/Soil_Experiment/testes_segmentacao/PBV29_Talita/tomoFiltered_masked_2048x2048x1964_16bit.raw"
# image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'uint16')
# image_binary = image.binarize(image_grayscale, dtype_out = 'uint16')
# img_num = 2
# print("fineshed reading big image!")

# IMAGE 3
# print("reading medium image...")
# xsize = 600
# ysize = 1520
# zsize = 1520
# path_grayscale = "../../example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw"
# path_binary = "../../example_images/binary/ILSIMG_600x1520x1520_16bits.raw"
# image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'uint16')
# image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
# img_num = 3
# print("fineshed reading medium image!")

# IMAGE 4
# print("reading big image...")
xsize = 2052
ysize = 2052
zsize = 2048

path_grayscale = "../../../../../../labs/tepui/home/camila.araujo/work/harpia/example_images/grayscale/Recon_2052x2052x2048_32bits.raw"
# path_binary = "../../../../../../../../labs/tepui/home/camila.araujo/work/harpia/example_images/binary/Recon_2052x2052x2048_16bits.raw"
image_grayscale = load(path_grayscale, xsize, ysize, zsize,'float32')[:100]
zsize = 100
# image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
img_num = 4
print("finished reading big image!")

#############
# Tests
#############

operations_filters = [
    {
        "name": "Gaussian Filter 3D grayscale",
        "skimage": gaussian,
        "skimage_param": {'mode': 'reflect'},
        "custom": gaussianFilter,
        "cucim": cucim_filters.gaussian,
    },
    {
        "name": "Mean Filter 3D grayscale",
        "skimage": None,
        "custom": meanFilter,
        "cucim": None,  # 2D only – no 3D support in cucim
    },
    {
        "name": "Median Filter 3D grayscale",
        "skimage": None,
        "custom": median,
        "cucim": cucim_filters.median,  # 2D support; 3D needs manual handling
    },
    {
        "name": "Log Filter 3D grayscale",
        "skimage": None,
        "custom": logFilter,
        "cucim": cucim_filters.laplace,  # Closest match to LoG; no direct LoG in cucim
    },
    {
        "name": "Unsharp Mask Filter 3D grayscale",
        "skimage": None,
        "custom": unsharpMaskFilter,
        "cucim": None  # Not available in cucim as of now
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
        "cucim": cucim_filters.sobel
    },
    {
        "name": "Prewitt Filter 3D grayscale",
        "skimage": prewitt,
        "custom": prewittFilter,
        "cucim": cucim_filters.prewitt
    },
    # {
    #     "name": "Anisotropic Diffusion Filter 3D grayscale",
    #     "skimage": None,
    #     "custom": anisotropic_diffusion3D,
    #     "cucim": None  # Not implemented in cucim
    # },
]


operations_thresholds = [
    {
        "name": "Threshold Niblack",
        "skimage": threshold_niblack,
        "custom": niblackThreshold,  
        "cucim": cucim_threshold_niblack,
    },
    {
        "name": "Threshold Sauvola",
        "skimage": threshold_sauvola,
        "custom": sauvolaThreshold,
        "cucim": cucim_threshold_sauvola,
    },
    {
        "name": "Threshold Mean",
        "skimage": threshold_mean,
        "custom": meanThreshold,
        "cucim": None,
    },
    #{
    #    "name": "Threshold Gaussian",
    #    "skimage": threshold_gaussian,
    #    "custom": gaussianThreshold,
    #    "cucim": lambda img, **kwargs: cucim_threshold_local(img, method='gaussian', **kwargs)
    #}
]

images_grayscale = [
    "float32",
    # "int32",
    # "uint32",
]

machine = 'harriet'
ngpus_values = [1]
gpuMemory_values = [0.1]
repetitions = 10

for ngpus in ngpus_values:
    for gpuMemory in gpuMemory_values:
        #csv_file = f"results_cucim/{machine}_{ngpus}gpu_{repetitions}reps_cython_results.csv"
        csv_file = f"{machine}_{ngpus}gpu_{repetitions}reps_cython_results.csv"

        for dtype in images_grayscale:
            image_input = image_grayscale.astype(dtype=dtype)
            for operation in operations_filters:
                # Attempt to run the test
                results_df = tests.run(
                    csv_file, image_input, operation, machine, ngpus, repetitions, gpuMemory, kernel = None)
                

for ngpus in ngpus_values:
    for gpuMemory in gpuMemory_values:
        csv_file = f"{machine}_{ngpus}gpu_{repetitions}reps_cython_results.csv"

        for dtype in images_grayscale:
            image_input = image_grayscale.astype(dtype=dtype)
            for operation in operations_thresholds:
                results_df = tests.run(
                    csv_file,
                    image_input,
                    operation,
                    machine,
                    ngpus,
                    repetitions,
                    gpuMemory,
                    kernel=None,
                )

print(f"The End!")