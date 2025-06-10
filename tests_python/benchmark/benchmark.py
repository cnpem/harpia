#!/usr/bin/env python3
import numpy as np                     # For array manipulation
from framework import image, tests
import cupy as cp

# Grayscale morphology operations
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
    gaussian)

from cucim.skimage import morphology as cucim_morph
from cucim.skimage import filters as cucim_filters

# Binary morphology operations
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening

# Workaround to allow importing harpia python module locally
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

# Custom morphology operations from harpia for binary images
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

# Custom morphology operations from harpia for grayscale images
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

import harpia
print(harpia.__file__)

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

#############
# Read Images
#############

# Instruction: Uncomment the image for which tests will be executed.

# IMAGE 1
print("reading small image...")
xsize = 190
ysize = 207
zsize_original = 100
zsize = 100
path_grayscale = "../../example_images/grayscale/crua_A_190x207x100_16b.raw"
path_binary = "../../example_images/binary/crua_A_190x207x100_16b.raw"
image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'uint16')
image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
img_num = 1
print("fineshed reading small image!")

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
# xsize = 2052
# ysize = 2052
# zsize = 2048

# path_grayscale = "../../../../../../../../labs/tepui/home/camila.araujo/work/harpia/example_images/grayscale/Recon_2052x2052x2048_32bits.raw"
# path_binary = "../../../../../../../../labs/tepui/home/camila.araujo/work/harpia/example_images/binary/Recon_2052x2052x2048_16bits.raw"
# image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'float32')
# image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
# img_num = 4
# print("fineshed reading big image!")

#Kernel
kernel = custum_kernel3D()

#############
# Tests
#############

operations_grayscale = [
    {
        "name": "Erosion 3D grayscale",
        "skimage": erosion,
        "custom": erosion_grayscale,
        "cucim": cucim_morph.erosion
    },
    {
        "name": "Dilation 3D grayscale",
        "skimage": dilation,
        "custom": dilation_grayscale,
        "cucim": cucim_morph.dilation
    },
    {
        "name": "Closing 3D grayscale",
        "skimage": closing,
        "custom": closing_grayscale,
        "cucim": cucim_morph.closing
    },
    {
        "name": "Opening 3D grayscale",
        "skimage": opening,
        "custom": opening_grayscale,
        "cucim": cucim_morph.opening
    },
    # {
    #     "name": "Geodesisc Erosion 3D grayscale",
    #     "skimage": None,
    #     "custom": geodesic_erosion_grayscale,
    #     "cucim": "geodesic_erosion_grayscale"
    # },
    # {
    #     "name": "Geodesisc Dilation 3D grayscale",
    #     "skimage": None,
    #     "custom": geodesic_dilation_grayscale,
    #     "cucim": "geodesic_dilation_grayscale"
    # },
    {
        "name": "Top Hat 3D grayscale",
        "skimage": white_tophat,
        "custom": top_hat,
        "cucim": cucim_morph.white_tophat
    },
    {
        "name": "Bottom Hat 3D grayscale",
        "skimage": black_tophat,
        "custom": bottom_hat,
        "cucim": cucim_morph.black_tophat
    },
]

operations_filters = [
    {
        "name": "Gaussian Filter 3D grayscale",
        "skimage": gaussian,
        "skimage_param":{'mode':'reflect'},
        "custom": gaussianFilter,
        "cucim": cucim_filters.gaussian,
    },
    # {
    #     "name": "Mean Filter 3D grayscale",
    #     "skimage": None,
    #     "custom": meanFilter,
    #     "cucim": "meanFilter"
    # },
    # {
    #     "name": "Log Filter 3D grayscale",
    #     "skimage": None,
    #     "custom": logFilter,
    #     "cucim": "logFilter"
    # },
    # {
    #     "name": "Unsharp Mask Filter 3D grayscale",
    #     "skimage": None,
    #     "custom": unsharpMaskFilter,
    #     "cucim": "unsharpMaskFilter"
    # },
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
    #     "cucim": "anisotropic_diffusion3D"
    # },
]

images_grayscale = [
    "float32",
    # "int32",
    # "uint32",
]

operations_binary = [
    {
        "name": "Erosion 3D binary",
        "skimage": erosion,
        "custom": erosion_binary,
        "cucim": cucim_morph.binary_erosion
    },
    {
        "name": "Dilation 3D binary",
        "skimage": dilation,
        "custom": dilation_binary,
        "cucim": cucim_morph.binary_dilation
    },
    {
        "name": "Closing 3D binary",
        "skimage": closing,
        "custom": closing_binary,
        "cucim": cucim_morph.binary_closing
    },
    {
        "name": "Opening 3D binary",
        "skimage": opening,
        "custom": opening_binary,
        "cucim": cucim_morph.binary_opening
    },
    {
        "name": "Smoothing 3D binary",
        "skimage": smooth_sk,
        "custom": smooth_binary,
        "cucim": smooth_cucim
    },
    # {
    #     "name": "Geodesisc Erosion 3D binary",
    #     "skimage": None,
    #     "custom": geodesic_erosion_binary,
    #     "cucim": "geodesic_erosion_binary"
    # },
    # {
    #     "name": "Geodesisc Dilation 3D binary",
    #     "skimage": None,
    #     "custom": geodesic_dilation_binary,
    #     "cucim": "geodesic_dilation_binary"
    # },
]

images_binary = [
    "int32",
    # "int16",
    # "uint16",
    # "uint32",
]

machine = 'harriet'
ngpus_values = [1]
gpuMemory_values = [0.4]
repetitions = 1

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
            for operation in operations_grayscale:
                # Attempt to run the test
                results_df = tests.run(
                    csv_file, image_input, operation, machine, ngpus, repetitions, gpuMemory, kernel
                )

        for dtype in images_binary:
            image_input = image_binary.astype(dtype=dtype)
            for operation in operations_binary:
                # Attempt to run the test
                results_df = tests.run(
                    csv_file, image_input, operation, machine, ngpus, repetitions, gpuMemory, kernel
                )

print(f"The End!")