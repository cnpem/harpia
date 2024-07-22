import numpy as np
import matplotlib.pyplot as plt

# workaround to allow importing harpia python module
import sys
sys.path.append('../../')
from harpia.morphology.operations_binary import erosionBinary


def contiguous(array: np.ndarray ) -> np.ndarray:
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array.astype(array.type()))

# Define the dimensions of the image
xsize = 355  # replace with the actual xsize
ysize = 321  # replace with the actual ysize
depth = 1

# Read the raw data from the file
raw_data = np.fromfile('../../example_images/binary/blobs_355x321x1_16b.raw', dtype=np.uint16)
raw_data = raw_data.astype(np.int32)
# Check the size of the data
if raw_data.size != xsize * ysize * depth:
    raise ValueError(f"Expected {xsize * ysize * depth} elements, but got {raw_data.size} elements.")

# Reshape the data into the correct dimensions
image = raw_data.reshape((ysize, xsize, depth))

# Create an output array
output_image = np.zeros_like(image)

# Define the kernel (example 3x3 kernel)
kernel = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
], dtype=np.int32)

# Define kernel and image sizes
kernel_xsize = 3
kernel_ysize = 3
kernel_zsize = 1
zsize = 1
flag_verbose = 0

kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1)

print(image.shape)
print(output_image.shape)
print(kernel.shape)

# Call the erosion_binary function
erosionBinary(image, output_image, kernel, kernel_xsize, kernel_ysize, kernel_zsize, xsize, ysize, zsize, flag_verbose)

#image = image.reshape((ysize, xsize))
#output_image = output_image.reshape((ysize, xsize))

# Plot the original and the processed images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(output_image, cmap='gray')
axes[1].set_title('Processed Image')
axes[1].axis('off')

plt.show()