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

machine = 'aida'
ngpus_values = [1]
gpuMemory_values = [0.1]
repetitions = 1
reps = 10
nslices = 2048

image_sets = [
    ("binary", images_binary, image_binary, operations.binary_cucim),
    ("grayscale", images_grayscale, image_grayscale, operations.grayscale_cucim)
]

for ngpus in ngpus_values:
    print("GPUs:", ngpus, "\n")
    #csv_file = f"results_cucim/{machine}_{ngpus}gpu_{repetitions}reps_cython_results.csv"
    csv_file = f"results_aida/{machine}_{ngpus}gpu_{repetitions}reps_cython_results_run2_cucim2048.csv"

    for gpuMemory in gpuMemory_values:
        print("\ngpuMemory:", gpuMemory,"\n")
        for image_type, dtypes, image, ops in image_sets:
            for dtype in dtypes:
                image_input = image.astype(dtype=dtype)
                #image_input = image_input[:nslices,:,:]  # Uncomment if needed
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
