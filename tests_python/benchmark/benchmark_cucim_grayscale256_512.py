#!/usr/bin/env python3
from framework import image, tests
from framework import operations

#############
# Read Images
#############

# Instruction: Uncomment the image for which tests will be executed.

# IMAGE 1
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

# IMAGE 4
print("reading big image...")
xsize = 2052
ysize = 2052
zsize = 2048

path_grayscale = "../../example_images/grayscale/Recon_2052x2052x2048_32bits.raw"
path_binary = "../../example_images/binary/Recon_2052x2052x2048_16bits.raw"
image_grayscale = image.load(path_grayscale, xsize, ysize, zsize,'float32')
image_binary = image.load(path_binary, xsize, ysize, zsize,'uint16')
img_num = 4
print("fineshed reading big image!")

#Kernel


#############
# Tests
#############

images_grayscale = [
    "float32",
    # "int32",
    # "uint32",
]


images_binary = [
    "int32",
    # "int16",
    # "uint16",
    # "uint32",
]

machine = 'mary'
ngpus = 1
gpuMemory = 0.1
repetitions = 1
reps = 30
nslices_values = [256, 512]

image_sets = [
#    ("binary", images_binary, image_binary, operations.binary_cucim),
    ("grayscale", images_grayscale, image_grayscale, operations.grayscale_cucim_no_threashold)
]

for nslices in nslices_values:
    for image_type, dtypes, image, ops in image_sets:
        csv_file = f"results_mary/{machine}_{reps}reps_cucim{nslices}_{image_type}.csv"
        print("Saving to file: ", csv_file)
        for dtype in dtypes:
            image_input = image.astype(dtype=dtype)
            image_input = image_input[:nslices,:,:]  # Uncomment if needed
            for _ in range(reps):
                for operation in ops:
                    kernel = operation["kernel"]
                    results_df = tests.run(
                        csv_file, 
                        image_input, 
                        operation, 
                        machine, 
                        ngpus, 
                        repetitions, 
                        gpuMemory, 
                        kernel
                    )

print(f"The End!")
