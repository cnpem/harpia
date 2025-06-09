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

# GRAYSCALE
operations_grayscale = [
    ("Erosion 3D grayscale", erosion, erosion_grayscale, cucim_morph.erosion),
    ("Dilation 3D grayscale", dilation, dilation_grayscale, cucim_morph.dilation),
    ("Closing 3D grayscale", closing, closing_grayscale, cucim_morph.closing),
    ("Opening 3D grayscale", opening, opening_grayscale, cucim_morph.opening),
#    ("Geodesisc Erosion 3D grayscale", None, geodesic_erosion_grayscale, "geodesic_erosion_grayscale"),
#    ("Geodesisc Dilation 3D grayscale", None, geodesic_dilation_grayscale, "geodesic_dilation_grayscale"),
    ("Top Hat 3D grayscale", white_tophat, top_hat, cucim_morph.white_tophat),
    ("Bottom Hat 3D grayscale", black_tophat, bottom_hat, cucim_morph.black_tophat),
]
operations_filters = [
    # ("Gaussian Filter 3D grayscale", None, gaussianFilter, "gaussianFilter"),
    # ("Mean Filter 3D grayscale", None, meanFilter, "meanFilter"),
    # ("Log Filter 3D grayscale", None, logFilter, "logFilter"),
    # ("Unsharp Mask Filter 3D grayscale", None, unsharpMaskFilter, "unsharpMaskFilter"),
    ("Sobel Filter 3D grayscale", sobel, sobelFilter, None),
    ("Prewitt Filter 3D grayscale", prewitt, prewittFilter, None),
    # ("Anisotropic Diffusion Filter 3D grayscale", None, anisotropic_diffusion3D, "anisotropic_diffusion3D"), #only runs in float images
]

images_grayscale = [
    #("int32", f"image{img_num}_int32_grayscale"),
    #("uint32", f"image{img_num}_uint32_grayscale"),
    ("float32", f"image{img_num}_float32_grayscale"),
   ]


# BINARY
operations_binary = [
    ("Erosion 3D binary", erosion, erosion_binary, cucim_morph.binary_erosion),
    ("Dilation 3D binary", dilation, dilation_binary, cucim_morph.binary_dilation),
    ("Closing 3D binary", closing, closing_binary, cucim_morph.binary_closing),
    ("Opening 3D binary", opening, opening_binary, cucim_morph.binary_opening),
    ("Smoothing 3D binary", smooth_sk, smooth_binary, smooth_cucim),
#    ("Geodesisc Erosion 3D binary", None, geodesic_erosion_binary, "geodesic_erosion_binary"),
#    ("Geodesisc Dilation 3D binary", None, geodesic_dilation_binary, "geodesic_dilation_binary"),
]

images_binary = [
    #("int16", f"image{img_num}_int16_binary"),
    #("uint16", f"image{img_num}_uint16_binary"),
    ("int32", f"image{img_num}_int32_binary"),
    #("uint32", f"image{img_num}_uint32_binary"),
   ]

machine = 'harriet'
ngpus_values = [1]
gpuMemory_values = [0.4]
repetitions = 1

for ngpus in ngpus_values:
    for gpuMemory in gpuMemory_values:
        csv_file = f"results_cucim/{machine}_{ngpus}gpu_{repetitions}reps_cython_results.csv"

        for img in images_grayscale:
            image_input = image_grayscale.astype(dtype=img[0])
            for operation in operations_filters:
                # Attempt to run the test
                results_df = tests.run_no_kernel(
                    csv_file, image_input, operation, machine, ngpus, repetitions, gpuMemory)
            for operation in operations_grayscale:
                # Attempt to run the test
                results_df = tests.run(
                    csv_file, image_input, operation, machine, ngpus, repetitions, gpuMemory, kernel
                )

        for img in images_binary:
            image_input = image_binary.astype(dtype=img[0])
            for operation in operations_binary:
                # Attempt to run the test
                results_df = tests.run(
                    csv_file, image_input, operation, machine, ngpus, repetitions, gpuMemory, kernel
                )

print(f"The End!")