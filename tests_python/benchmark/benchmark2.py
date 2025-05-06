#!/usr/bin/env python3
import numpy as np                     # For array manipulation
from framework import image, tests
import csv
import h5py

# Grayscale morphology operations
from skimage.morphology import (
    erosion, dilation, closing, opening,
    white_tophat, black_tophat, reconstruction
)

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

#############
# Read Images
#############

# Instruction: Uncomment the image for which tests will be executed.

# # IMAGE 1
# print("reading small image...")
# xsize = 190
# ysize = 207
# zsize_original = 100
# zsize = 100
# path_grayscale = "../../example_images/grayscale/crua_A_190x207x100_16b.raw"
# path_binary = "../../example_images/binary/crua_A_190x207x100_16b.raw"
# image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'uint16')
# image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
# img_num = 1
# print("fineshed reading small image!")

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

# # IMAGE 3
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
print("reading big image...")
xsize = 2052
ysize = 2052
zsize = 2048

path_grayscale = "../../../../../../../../labs/tepui/home/camila.araujo/work/harpia/example_images/grayscale/Recon_2052x2052x2048_32bits.raw"
path_binary = "../../../../../../../../labs/tepui/home/camila.araujo/work/harpia/example_images/binary/Recon_2052x2052x2048_16bits.raw"
image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'float32')
image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
img_num = 4
print("fineshed reading big image!")

#Kernel
kernel = custum_kernel3D()

#############
# Tests
#############

machine = 'harriet'
ngpus = 3
gpuMemory = 0.2
repetitions = 11

# GRAYSCALE
operations = [
    ("Erosion 3D grayscale", None, erosion_grayscale, "erosion_grayscale"),
    ("Dilation 3D grayscale", None, dilation_grayscale, "dilation_grayscale"),
    ("Closing 3D grayscale", None, closing_grayscale, "closing_grayscale"),
    ("Opening 3D grayscale", None, opening_grayscale, "opening_grayscale"),
]

images = [
    ("int32", f"image{img_num}_int32_grayscale"),
    ("uint32", f"image{img_num}_uint32_grayscale"),
    ("float32", f"image{img_num}_float32_grayscale"),
   ]

#Iterate over operations and images
for img in images:
    image_input = image_grayscale.astype(dtype=img[0])
    for operation in operations:
        csv_file = f"results{img_num}/{machine}_{ngpus}gpu_{operation[3]}_{img[1]}.csv"
        # Attempt to run the test
        results_df = tests.run(
            csv_file, image_input, operation, machine, ngpus, True, repetitions, gpuMemory, kernel
        )

# BINARY
operations = [
    ("Erosion 3D binary", None, erosion_binary, "erosion_binary"),
    ("Dilation 3D binary", None, dilation_binary, "dilation_binary"),
    ("Closing 3D binary", None, closing_binary, "closing_binary"),
    ("Opening 3D binary", None, opening_binary, "opening_binary"),
    ("Smoothing 3D binary", None, smooth_binary, "smooth_binary"),
]

images_binary = [
    ("int16", f"image{img_num}_int16_binary"),
    ("uint16", f"image{img_num}_uint16_binary"),
    ("int32", f"image{img_num}_int32_binary"),
    ("uint32", f"image{img_num}_uint32_binary"),
   ]

#Iterate over operations and images
for img in images_binary:
    image_input = image_binary.astype(dtype=img[0])
    for operation in operations:
        csv_file = f"results{img_num}/{machine}_{ngpus}gpu_{operation[3]}_{img[1]}.csv"
        # Attempt to run the test
        results_df = tests.run(
            csv_file, image_input, operation, machine, ngpus, True, repetitions, gpuMemory, kernel
        )

print(f"The End!")
