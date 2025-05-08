import timeit                           # For timing the function
import numpy as np                     # For array manipulation
import matplotlib.pyplot as plt         # For plotting images

def time_module_only(csv_data, machine, module_func, image, plot=False, show=False, operation="",
                     slice_num=0, figsize=(18, 6), save_path=None, repetitions=1, gpuMemory=None, 
                     ngpus = -1, *args,  **kwargs):
    fontsize = 18
    times = []

    # Perform the function multiple times to average timing, ignoring the first run
    if(gpuMemory):
        module_output = module_func(image, *args, gpuMemory= gpuMemory, ngpus = ngpus, **kwargs) #warm up
        for _ in range(repetitions):
            start = timeit.default_timer()
            module_output = module_func(image, *args, gpuMemory= gpuMemory, ngpus = ngpus, **kwargs)
            times.append(timeit.default_timer() - start)
    else:
        module_output = module_func(image, *args, ngpus = ngpus, **kwargs) #warm up
        for _ in range(repetitions):
            gpuMemory = 0
            start = timeit.default_timer()
            module_output = module_func(image, *args, ngpus = ngpus, **kwargs)
            times.append(timeit.default_timer() - start)

    # Calculate the mean time (ignoring the first warm-up run if repetitions > 1)
    module_time = np.mean(times)

    # Get the image data type, size, and dimensions
    image_dtype = str(image.dtype)
    image_size_mb = round(image.nbytes / (1024 ** 2),1)
    image_shape = (image.shape[2], image.shape[1], image.shape[0]) #compatible with c++ shape

    # Add timing information, data type, size, and dimensions to CSV data
    csv_data.append({
        'Operation':  module_func.__name__ if not operation else operation,
        'Machine': machine,
        'Gpus': ngpus,
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