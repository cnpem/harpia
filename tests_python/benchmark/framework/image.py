import numpy as np                     # For array manipulation
import matplotlib.pyplot as plt         # For plotting image

def contiguous(array: np.ndarray) -> np.ndarray:
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array.astype(array.type()))
    
def load(path, xsize, ysize, zsize, dtype):
    img = np.fromfile(path, dtype=dtype)
    img = img.reshape((zsize, ysize, xsize))
    contiguous(img)
    return img

def binarize(data, plot=False, dtype_out='int32'):
    zsize, _, _ = data.shape  # Get dimensions
    binarized_data = np.empty_like(data, dtype = dtype_out)  # Prepare output array of same shape
    contiguous(binarized_data)

    for slice_idx in range(zsize):
        slice_data = data[slice_idx, :, :]

        # Find min and max for the current slice
        min_val = slice_data.min()
        max_val = slice_data.max()

        # Compute threshold
        threshold = (max_val + min_val) // 2

        # Apply threshold to the slice to create a binary image
        binarized_slice = np.where(slice_data >= threshold, 1, 0)

        # Store the binarized slice in the output array
        binarized_data[slice_idx, :, :] = binarized_slice

    # Plot the first slice if plot flag is True
    if plot:
        plt.figure(figsize=(10, 4))

        # Plot original first slice
        plt.subplot(1, 2, 1)
        plt.imshow(data[0, :, :], cmap='gray')
        plt.title('Original First Slice')
        plt.axis('off')

        # Plot binarized first slice
        plt.subplot(1, 2, 2)
        plt.imshow(binarized_data[0, :, :], cmap='gray')
        plt.title('Binarized First Slice')
        plt.axis('off')

        plt.show()

    return binarized_data

def binarize_by_slice(data, plot=False, dtype_out='int32'):
    zsize, _, _ = data.shape  # Get dimensions

    # Compute min and max for each slice
    min_vals = data.min(axis=(1, 2), keepdims=True)  # (zsize, 1, 1)
    max_vals = data.max(axis=(1, 2), keepdims=True)  # (zsize, 1, 1)

    # Compute thresholds for each slice
    thresholds = (max_vals + min_vals) // 2  # Broadcasts to (zsize, 1, 1)

    # Binarize each slice using vectorized comparison
    binarized_data = np.where(data >= thresholds, 1, 0).astype(dtype_out)

    # Plot the first slice if plot flag is True
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        # Plot original first slice
        plt.subplot(1, 2, 1)
        plt.imshow(data[0, :, :], cmap='gray')
        plt.title('Original First Slice')
        plt.axis('off')

        # Plot binarized first slice
        plt.subplot(1, 2, 2)
        plt.imshow(binarized_data[0, :, :], cmap='gray')
        plt.title('Binarized First Slice')
        plt.axis('off')

        plt.show()

    return binarized_data