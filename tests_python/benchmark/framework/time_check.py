import timeit                           # For timing the function
import numpy as np                     # For array manipulation
import matplotlib.pyplot as plt         # For plotting images

# Merged function to time, compare, and plot results with MSE instead of accuracy
def time_compare_and_plot(csv_data, machine, module_func, skimage_func, image, kernel, 
                          plot=False, show=False, operation="", framework="", slice_num=0, 
                          figsize=(18, 6), save_path=None, repetitions=1, gpuMemory =0.4,
                          ngpus = -1, *args, **kwargs):
    fontsize = 18
    module_times = []
    skimage_times = []

    # Time the module function multiple times
    module_output = module_func(image, kernel, *args, gpuMemory= gpuMemory, **kwargs) #warm up run
    for _ in range(repetitions):
        start = timeit.default_timer()
        module_output = module_func(image, kernel, *args, gpuMemory= gpuMemory, ngpus = ngpus, **kwargs)
        module_times.append(timeit.default_timer() - start)

    # Time the scikit-image function multiple times
    skimage_output = skimage_func(image, kernel) #warm up run
    for _ in range(repetitions):
        start = timeit.default_timer()
        skimage_output = skimage_func(image, kernel)
        skimage_times.append(timeit.default_timer() - start)

    # Calculate mean times
    module_time = np.mean(module_times)
    skimage_time = np.mean(skimage_times)

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
    image_shape = (image.shape[2], image.shape[1], image.shape[0]) #compatible with c++ shape

    # Add timing, MSE, and image details to CSV data
    csv_data.append({
        'Operation':  module_func.__name__ if not operation else operation,
        'Machine': machine,
        'Gpus': ngpus,
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
        print(f"Image Data Type: {image_dtype}")
        print(f"Image Size: {image_size_mb} MB")
        print(f"Image Dimensions: {image_shape}")