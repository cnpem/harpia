#!/usr/bin/env python3
import numpy as np                     # For array manipulation
import pandas as pd                    # For data handling
import timeit                           # For timing the function
import matplotlib.pyplot as plt         # For plotting images

# Grayscale morphology operations
from skimage.morphology import (       
    erosion, dilation, closing, opening, 
    white_tophat, black_tophat, reconstruction
)

# Binary morphology operations
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening

# Workaround to allow importing harpia python module
import sys
sys.path.append("../../")

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
############
#Framework
############

def binary_smooth(image, selem):
    result = binary_opening(image, selem)
    result = binary_closing(result, selem)
    return result

def contiguous(array: np.ndarray) -> np.ndarray:
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array.astype(array.type()))
        
def load_image(path, xsize, ysize, zsize, dtype, dtype_out):
    img = np.fromfile(path, dtype=dtype)
    img = img.reshape((zsize, ysize, xsize))
    img = img.astype(dtype = dtype_out)
    contiguous(img)
    return img
    
def custum_kernel3D():
    kernel_2d = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
    # Stack the 2D kernel to form a 3D kernel (3 layers)
    kernel_3d = np.stack([kernel_2d, kernel_2d, kernel_2d])
    return kernel_3d
    
def time_module_only(csv_data,  hardware, machine, module_func, image, plot=False, show=False, operation="",
                     slice_num=0, figsize=(18, 6), save_path=None, repetitions=1, gpuMemory=None, *args,  **kwargs):
    fontsize = 18
    times = []

    # Perform the function multiple times to average timing, ignoring the first run
    for _ in range(repetitions):
        if(gpuMemory):
            start = timeit.default_timer()
            module_output = module_func(image, *args, gpuMemory= gpuMemory, **kwargs)
            times.append(timeit.default_timer() - start)
        else:
            gpuMemory = 0
            start = timeit.default_timer()
            module_output = module_func(image, *args, **kwargs)
            times.append(timeit.default_timer() - start)

    # Calculate the mean time (ignoring the first warm-up run if repetitions > 1)
    if repetitions > 1:
        module_time = np.mean(times[1:])
        repetitions = repetitions-1
    else:
        module_time = times[0]

    # Get the image data type, size, and dimensions
    image_dtype = str(image.dtype)
    image_size_bytes = image.nbytes
    image_size_mb = round(image.nbytes / (1024 ** 2),1)
    image_shape = image.shape

    # Add timing information, data type, size, and dimensions to CSV data
    csv_data.append({
        'Operation':  module_func.__name__ if not operation else operation,
        'Machine': machine,
        'Gpus': hardware,
        'gpuMemory': gpuMemory, 
        'Module Time (s)': module_time,
        'Scikit-Image Time (s)': 'N/A',
        'Time Ratio': 'N/A',
        'Repetitions' : repetitions,
        'Mean Squared Error': 'N/A',
        'Pixel Accuracy (%)': 'N/A',
        'Image Data Type': image_dtype,
        'Image Size (MB)': image_size_mb,
        'Image Dimensions': image_shape
    })
    # Plot results if specified
    if plot:
        if len(image.shape) == 3:
            original_slice = image[slice_num, :, :]
            slice_module = module_output[slice_num, :, :]

        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.imshow(original_slice, cmap='gray')                  
        plt.title("Original Image", fontsize=fontsize)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(slice_module, cmap='gray')
        plt.title(f"Annotat3d {operation}", fontsize=fontsize)
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # Print timing information
    if show:
        print(f"Operation: {module_func.__name__ if not operation else operation}")
        print(f"Module Time: {module_time:.4f} seconds")
        print(f"Image Data Type: {image_dtype}")
        print(f"Image Size: {image_size_mb} MB")
        print(f"Image Dimensions: {image_shape}")


# Merged function to time, compare, and plot results with MSE instead of accuracy
def time_compare_and_plot(csv_data, hardware, machine, module_func, skimage_func, image, kernel, plot=False, show=False,
                          operation="", framework="", slice_num=0, figsize=(18, 6), save_path=None, repetitions=1, gpuMemory =0.4,
                          *args, **kwargs):
    fontsize = 18
    module_times = []
    skimage_times = []

    # Time the module function multiple times
    for _ in range(repetitions):
        start = timeit.default_timer()
        module_output = module_func(image, kernel, *args, gpuMemory= gpuMemory, **kwargs)
        module_times.append(timeit.default_timer() - start)

    # Time the scikit-image function multiple times
    for _ in range(repetitions):
        start = timeit.default_timer()
        skimage_output = skimage_func(image, kernel)
        skimage_times.append(timeit.default_timer() - start)

    # Calculate mean times, ignoring the first run if repetitions > 1
    if repetitions > 1:
        module_time = np.mean(module_times[1:])
        skimage_time = np.mean(skimage_times[1:])
        repetitions = repetitions-1
    else:
        module_time = module_times[0]
        skimage_time = skimage_times[0]

    # Calculate Mean Squared Error
    mse = np.mean((skimage_output.astype(np.float32) - module_output.astype(np.float32)) ** 2)
    bitwise_diff = np.abs(skimage_output.astype(np.int32) - module_output.astype(np.int32))
    total_pixels = np.prod(image.shape)
    num_diff_pixels = np.count_nonzero(bitwise_diff)
    pixel_accuracy = ((total_pixels - num_diff_pixels) / total_pixels) * 100
    faster = round(skimage_time/module_time, 1)

    # Get the image data type, size, and dimensions
    image_dtype = str(image.dtype)
    image_size_mb = round(image.nbytes/(1024 ** 2),1)
    image_shape = image.shape

    # Add timing, MSE, and image details to CSV data
    csv_data.append({
        'Operation':  module_func.__name__ if not operation else operation,
        'Machine': machine,
        'Gpus': hardware,
        'gpuMemory': gpuMemory, 
        'Module Time (s)': module_time,
        'Scikit-Image Time (s)': skimage_time,
        'Time Ratio': faster,
        'Repetitions' : repetitions,
        'Mean Squared Error': mse,
        'Pixel Accuracy (%)': pixel_accuracy,
        'Image Data Type': image_dtype,
        'Image Size (MB)': image_size_mb,
        'Image Dimensions': image_shape
    })

    # Plot results if specified
    if plot:
        if len(image.shape) == 3:
            original_slice = image[slice_num, :, :]
            slice_skimage = skimage_output[slice_num, :, :]
            slice_module = module_output[slice_num, :, :]

        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.imshow(original_slice, cmap='gray')
        plt.title("Original Image", fontsize=fontsize)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(slice_skimage, cmap='gray')
        plt.title(f"{framework} {operation}", fontsize=fontsize)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(slice_module, cmap='gray')
        plt.title(f"Annotat3d {operation}", fontsize=fontsize)
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # Print statistics
    if show:
        print(f"Operation: {module_func.__name__ if not operation else operation}")
        print(f"Module Time: {module_time:.4f} seconds")
        print(f"Pixel Accuracy: {pixel_accuracy:.2f}%")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Difference value: {mean_diff:.2f} ± {std_diff:.2f}")
        print(f"Image Data Type: {image_dtype}")
        print(f"Image Size: {image_size_mb} MB")
        print(f"Image Dimensions: {image_shape}")
    
def binarize_image(data, plot=False, dtype_out='int32'):
    zsize, ysize, xsize = data.shape  # Get dimensions
    binarized_data = np.empty_like(data, dtype = dtype_out)  # Prepare output array of same shape

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

############
#Read Images
############

print("Reading big image...")
# Big image
xsize = 2048
ysize = 2048
zsize = 1964
path = "../../../../../../../../beamlines/mogno/proposals/20180217/data/Soil_Experiment/testes_segmentacao/PBV29_Talita/tomoFiltered_masked_2048x2048x1964_16bit.raw"
image2_uint32_gray = load_image(path, xsize, ysize, zsize,'int16', 'uint32')
image2_int32_gray = load_image(path, xsize, ysize, zsize,'int16', 'int32')
image2_float32_gray = load_image(path, xsize, ysize, zsize,'int16', 'float32')

image2_int32_binary = binarize_image(image2_int32_gray, dtype_out = 'int32')
image2_uint32_binary = binarize_image(image2_int32_gray, dtype_out = 'uint32')
image2_int16_binary = binarize_image(image2_int32_gray, dtype_out = 'int16')
image2_uint16_binary = binarize_image(image2_int32_gray, dtype_out = 'uint16')
image2_int8_binary = binarize_image(image2_int32_gray, dtype_out = 'int8')
image2_uint8_binary = binarize_image(image2_int32_gray, dtype_out = 'uint8')

print("Reading small image...")
#Small
xsize = 190
ysize = 207
zsize_original = 100
zsize = 100
path = "../../example_images/grayscale/crua_A_190x207x100_16b.raw"

image1_uint32_gray = load_image(path, xsize, ysize, zsize,'uint16', 'uint32')
image1_int32_gray = load_image(path, xsize, ysize, zsize,'uint16', 'int32')
image1_float32_gray = load_image(path, xsize, ysize, zsize,'uint16', 'float32')

image1_int32_binary = binarize_image(image1_int32_gray, dtype_out = 'int32')
image1_uint32_binary = binarize_image(image1_int32_gray, dtype_out = 'uint32')
image1_int16_binary = binarize_image(image1_int32_gray, dtype_out = 'int16')
image1_uint16_binary = binarize_image(image1_int32_gray, dtype_out = 'uint16')
image1_int8_binary = binarize_image(image1_int32_gray, dtype_out = 'int8')
image1_uint8_binary = binarize_image(image1_int32_gray, dtype_out = 'uint8')

#Kernel
kernel = custum_kernel3D()

############
#Tests Framework
############

def run_tests(csv_file, images, operations, ngpus, gpu_flag, repetitions, gpuMemory):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    for img in images:
        print("")
        print(img.shape)
        for operation_name, skimage_func, harpia_func in operations:
            # Call the timing and comparison function
            print(operation_name)
            if(skimage_func):
                time_compare_and_plot(
                    csv_data=csv_data,
                    hardware=ngpus,
                    machine=machine,
                    gpuMemory=gpuMemory,
                    module_func=harpia_func,
                    skimage_func=skimage_func,
                    image=img,
                    kernel=kernel,
                    plot=False,
                    show=False,
                    operation=operation_name,
                    framework="scikit",
                    slice_num=0,
                    figsize=(18, 6),
                    save_path=f"plots/{operation_name.replace(' ', '_')}_{str(img.dtype)}.png",
                    repetitions=repetitions,
                    gpu=gpu_flag
                )
            else:
                print(operation_name)
                time_module_only(
                    csv_data=csv_data,
                    hardware=ngpus,
                    machine=machine,
                    gpuMemory=gpuMemory,
                    module_func=harpia_func,
                    image=img,
                    kernel=kernel,
                    plot=False,
                    show=False,
                    operation=operation_name,
                    slice_num=0,
                    figsize=(18, 6),
                    save_path=f"plots/{operation_name.replace(' ', '_')}_{str(img.dtype)}.png",
                    repetitions=repetitions,
                    gpu=gpu_flag
                )
    print('\nFinish Tests!')
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv(csv_file, index=False)
    return results_df

def run_tests_no_kernel(csv_file, images, operations, ngpus, gpu_flag, repetitions, gpuMemory):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    for img in images:
        print("")
        print(img.shape)
        for operation_name, skimage_func, harpia_func in operations:
            # Call the timing and comparison function
            print(operation_name)
            if(skimage_func):
                time_compare_and_plot(
                    csv_data=csv_data,
                    hardware=ngpus,
                    machine=machine,
                    gpuMemory=gpuMemory,
                    module_func=harpia_func,
                    skimage_func=skimage_func,
                    image=img,
                    plot=False,
                    show=False,
                    operation=operation_name,
                    framework="scikit",
                    slice_num=0,
                    figsize=(18, 6),
                    save_path=f"plots/{operation_name.replace(' ', '_')}_{str(img.dtype)}.png",
                    repetitions=repetitions,
                    gpu=gpu_flag
                )
            else:
                print(operation_name)
                time_module_only(
                    csv_data=csv_data,
                    hardware=ngpus,
                    machine=machine,
                    gpuMemory=gpuMemory,
                    module_func=harpia_func,
                    image=img,
                    plot=False,
                    show=False,
                    operation=operation_name,
                    slice_num=0,
                    figsize=(18, 6),
                    save_path=f"plots/{operation_name.replace(' ', '_')}_{str(img.dtype)}.png",
                    repetitions=repetitions,
                    gpu=gpu_flag
                )
    print('\nFinish Tests!')
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv(csv_file, index=False)
    return results_df

############
#Tests
############

machine = 'harriet'
gpuMemory = 0.41
repetitions = 11
ngpus = 1

# BINARY 1 BIG IMAGE

operations = [
    ("Erosion 3D binary", None, erosion_binary),
]
images = [image2_int32_binary,image2_uint32_binary,image2_int16_binary,image2_uint16_binary,image2_int8_binary,image2_uint8_binary]
csv_file = f"benchmark_morph_binary_erosion_big_{ngpus}gpu.csv"

results_df = run_tests(csv_file, images, operations, ngpus, True, repetitions, gpuMemory = 0.41)

# BINARY 3 BIG IMAGE

# operations = [
#     ("Erosion 3D binary", binary_erosion, erosion_binary),
#     ("Opening 3D binary", binary_opening, opening_binary),
#     ("Smoothing 3D binary", binary_smooth, smooth_binary),
#     ("Dilation 3D binary", binary_dilation, dilation_binary),
#     ("Closing 3D binary", binary_closing, closing_binary),
# ]
# images = [image2_int32_binary,image2_uint32_binary,image2_int16_binary,image2_uint16_binary,image2_int8_binary,image2_uint8_binary]
# csv_file = f"benchmark_morph_binary3_big_{ngpus}gpu.csv"

# results_df = run_tests(csv_file, images, operations, ngpus, True, repetitions, gpuMemory = 0.41)

