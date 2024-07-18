import numpy as np
import matplotlib.pyplot as plt

# workaround to allow importing harpia python module
import sys
sys.path.append('../../')
from harpia.morphology.operations_grayscale import erosion_grayscale

def contiguous(array: np.ndarray ) -> np.ndarray:
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array.astype(array.type()))
    return array
    

# Define the dimensions of the image
xsize = 190  # replace with the actual xsize
ysize = 207  # replace with the actual ysize
zsize = 100

# Read the raw data from the file
raw_data = np.fromfile('../../example_images/grayscale/crua_A_190x207x100_16b.raw', dtype=np.uint16)
raw_data = raw_data.astype(np.float32)
# Check the size of the data
if raw_data.size != xsize * ysize * zsize:
    raise ValueError(f"Expected {xsize * ysize * zsize} elements, but got {raw_data.size} elements.")

# Reshape the data into the correct dimensions
image = raw_data.reshape((zsize, ysize, xsize))

# Create an output array
output_image = np.zeros_like(image)

# Define the kernel (example 3x3 kernel)
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.int32)

# Define kernel and image sizes
kernel_xsize = 3
kernel_ysize = 3
kernel_zsize = 3

block_xsize = 16  # example block size
block_ysize = 16  # example block size
block_zsize = 1
flag_verbose = 0

kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1)

print(image.shape)
print(output_image.shape)
print(kernel.shape)

# Call the erosion_binary function
erosion_grayscale(contiguous(image), contiguous(output_image), kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, block_xsize, block_ysize, block_zsize, flag_verbose)

# Plot the original and the processed images
slice = 0
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image[:,:,slice], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(output_image[:,:,slice], cmap='gray')
axes[1].set_title('Processed Image')
axes[1].axis('off')

plt.show()