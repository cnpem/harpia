#!/usr/bin/env python3
import numpy as np                     # For array manipulation
from framework import image, tests
import csv

# Grayscale morphology operations
from skimage.morphology import (       
    erosion, dilation, closing, opening, 
    white_tophat, black_tophat, reconstruction
)

# Binary morphology operations
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening

# Workaround to allow importing harpia python module
import sys
import os
# sys.path.append("../../")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))) # read local harpia
# Get the parent directory and add it to sys.path
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# sys.path.append(parent_dir)

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

# def binary_smooth(image, selem):
#     result = binary_opening(image, selem)
#     result = binary_closing(result, selem)
#     return result

############
#Read Images
############

# # IMAGE 1
# print("reading small image...")
# xsize = 190
# ysize = 207
# zsize_original = 100
# zsize = 100
# path = "../../example_images/grayscale/crua_A_190x207x100_16b.raw"
# image_uint32_gray = image.load(path, xsize, ysize, zsize,'uint16', 'uint32')
# image_int32_gray = image.load(path, xsize, ysize, zsize,'uint16', 'int32')
# image_float32_gray = image.load(path, xsize, ysize, zsize,'uint16', 'float32')
# img_num = 1
# print("fineshed reading small image!")

# # IMAGE 2
print("reading big image...")
xsize = 2048
ysize = 2048
zsize = 1964
path = "../../../../../../../../beamlines/mogno/proposals/20180217/data/Soil_Experiment/testes_segmentacao/PBV29_Talita/tomoFiltered_masked_2048x2048x1964_16bit.raw"
image_uint32_gray = image.load(path, xsize, ysize, zsize,'int16', 'uint32')
image_int32_gray = image.load(path, xsize, ysize, zsize,'int16', 'int32')
image_float32_gray = image.load(path, xsize, ysize, zsize,'int16', 'float32')
img_num = 2
print("fineshed reading big image!")

# IMAGE 3
# print("reading medium image...")
# xsize = 600
# ysize = 1520
# zsize = 1520
# path = "../../example_images/grayscale/ILSIMG_600x1520x1520_16bits.raw"
# image_uint32_gray = image.load(path, xsize, ysize, zsize,'int16', 'uint32')
# image_int32_gray = image.load(path, xsize, ysize, zsize,'int16', 'int32')
# image_float32_gray = image.load(path, xsize, ysize, zsize,'int16', 'float32')
# img_num = 3
# print("fineshed reading medium image!")


#Kernel
kernel = custum_kernel3D()

############
#Tests
############

machine = 'harriet'
ngpus = 1
gpuMemory = 0.41
repetitions = 11

# BINARY 1 BIG IMAGE
operations = [
    ("Smoothing 3D binary", None, smooth_binary, "smooth_binary"),
    ("Closing 3D binary", None, closing_binary, "closing_binary"),
    ("Opening 3D binary", None, opening_binary, "opening_binary"),
    ("Erosion 3D binary", None, erosion_binary, "erosion_binary"),
    ("Dilation 3D binary", None, dilation_binary, "dilation_binary"),
]

# BINARY 1 BIG IMAGE
images = [
    ("int8", f"image{img_num}_int8_binary"),
    ("uint8", f"image{img_num}_uint8_binary"),
    ("int16", f"image{img_num}_int16_binary"),
    ("uint16", f"image{img_num}_uint16_binary"),    
    ("int32", f"image{img_num}_int32_binary"),
    ("uint32", f"image{img_num}_uint32_binary"),
   ]

#Iterate over operations and images
for img in images:
    binerized_image = image.binarize(image_int32_gray, dtype_out = img[0])
    for operation in operations:
        csv_file = f"results{img_num}/{machine}_{ngpus}gpu_{operation[3]}_{img[1]}.csv"
        # Attempt to run the test
        results_df = tests.run(
            csv_file, binerized_image, operation, machine, ngpus, True, repetitions, gpuMemory, kernel
        )

# BINARY 1 SMALL IMAGE
# images = [
#     ("int32", "image1_int32_binary"),
#     ("uint32", "image1_uint32_binary"),
#     ("int16", "image1_int16_binary"),
#     ("uint16", "image1_uint16_binary"),    
#     ("int8", "image1_int8_binary"),
#     ("uint8", "image1_uint8_binary"),
# ]

# # Iterate over operations and images
# for img in images:
#     binerized_image = image.binarize(image1_int32_gray, dtype_out = img[0])
#     for operation in operations:
#         csv_file = f"results2/{machine}_{ngpus}gpu_{operation[3]}_{img[1]}.csv"
#         # Attempt to run the test
#         results_df = tests.run(
#             csv_file, binerized_image, operation, machine, ngpus, True, repetitions, gpuMemory, kernel
#         )
    

print(f"The End!")